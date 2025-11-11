"""
Main OCR pipeline orchestration.
"""
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .config import Config
from .preprocessing import PDFProcessor, ImageEnhancer
from .ocr_engines import (
    TesseractEngine,
    GoogleVisionEngine,
    AWSTextractEngine,
    AzureVisionEngine
)
from .reconciliation import LLMReconciler
from .reconciliation.llm_providers import OpenAIProvider, AnthropicProvider, GeminiProvider
from .metadata import MetadataExtractor
from .storage import JSONStorage


class OCRPipeline:
    """Complete OCR pipeline for historical documents."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize OCR pipeline.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()

        # Initialize components
        self._init_preprocessing()
        self._init_ocr_engines()
        self._init_reconciliation()
        self._init_metadata_extraction()
        self._init_storage()

    def _init_preprocessing(self):
        """Initialize preprocessing components."""
        preproc_config = self.config.get_preprocessing_config()

        self.pdf_processor = PDFProcessor(dpi=300)
        self.image_enhancer = ImageEnhancer(**preproc_config)

    def _init_ocr_engines(self):
        """Initialize OCR engines based on configuration."""
        enabled_engines = self.config.get_ocr_engines()
        api_keys = self.config.get_api_keys()

        self.ocr_engines = []

        for engine_name in enabled_engines:
            if engine_name == 'tesseract':
                engine = TesseractEngine()
                if engine.is_available():
                    self.ocr_engines.append(engine)
                else:
                    print(f"Warning: Tesseract not available, skipping")

            elif engine_name == 'google_vision':
                engine = GoogleVisionEngine(
                    credentials_path=api_keys['google_vision']
                )
                if engine.is_available():
                    self.ocr_engines.append(engine)
                else:
                    print(f"Warning: Google Vision not available, skipping")

            elif engine_name == 'aws_textract':
                engine = AWSTextractEngine(
                    aws_access_key_id=api_keys['aws_access_key_id'],
                    aws_secret_access_key=api_keys['aws_secret_access_key'],
                    region_name=api_keys['aws_region']
                )
                if engine.is_available():
                    self.ocr_engines.append(engine)
                else:
                    print(f"Warning: AWS Textract not available, skipping")

            elif engine_name == 'azure_vision':
                engine = AzureVisionEngine(
                    subscription_key=api_keys['azure_key'],
                    endpoint=api_keys['azure_endpoint']
                )
                if engine.is_available():
                    self.ocr_engines.append(engine)
                else:
                    print(f"Warning: Azure Vision not available, skipping")

        if not self.ocr_engines:
            raise RuntimeError("No OCR engines available!")

        print(f"Initialized {len(self.ocr_engines)} OCR engine(s): "
              f"{[e.name for e in self.ocr_engines]}")

    def _init_reconciliation(self):
        """Initialize LLM reconciliation."""
        llm_config = self.config.get_llm_config()
        recon_config = self.config.get_reconciliation_config()
        api_keys = self.config.get_api_keys()

        provider_name = llm_config['provider']

        # Create LLM provider
        if provider_name == 'openai':
            provider = OpenAIProvider(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
                api_key=api_keys['openai_key']
            )
        elif provider_name == 'anthropic':
            provider = AnthropicProvider(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
                api_key=api_keys['anthropic_key']
            )
        elif provider_name == 'gemini':
            provider = GeminiProvider(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
                api_key=api_keys['gemini_key']
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

        if not provider.is_available():
            raise RuntimeError(f"LLM provider {provider_name} not available!")

        self.reconciler = LLMReconciler(
            llm_provider=provider,
            flag_threshold=recon_config['flag_threshold'],
            min_engines_required=recon_config['min_engines_required']
        )

        print(f"Initialized LLM reconciliation with {provider_name} ({llm_config['model']})")

    def _init_metadata_extraction(self):
        """Initialize metadata extraction."""
        self.metadata_extractor = MetadataExtractor()

    def _init_storage(self):
        """Initialize storage."""
        storage_config = self.config.get_storage_config()

        self.storage = JSONStorage(
            output_dir=storage_config['output_dir'],
            keep_intermediate=storage_config['keep_intermediate']
        )

    def process_document(
        self,
        pdf_path: str,
        document_name: Optional[str] = None,
        progress: bool = True
    ) -> str:
        """
        Process a complete PDF document.

        Args:
            pdf_path: Path to PDF file
            document_name: Optional name (uses filename if not provided)
            progress: Show progress bars

        Returns:
            Path to output JSON file
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if document_name is None:
            document_name = pdf_path.stem

        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}\n")

        # Step 1: Extract PDF info and convert to images
        print("Step 1: Converting PDF to images...")
        pdf_info, page_images = self.pdf_processor.process_pdf(str(pdf_path))

        # Step 2: Enhance images
        print("Step 2: Enhancing images...")
        enhanced_images = []
        for image, page_num in tqdm(page_images, desc="Enhancing", disable=not progress):
            enhanced = self.image_enhancer.enhance(image)
            enhanced_images.append((enhanced, page_num))

        # Step 3: Run OCR with all engines
        print(f"Step 3: Running OCR with {len(self.ocr_engines)} engine(s)...")
        all_ocr_results = []  # List of lists (one per page)

        for image, page_num in tqdm(enhanced_images, desc="OCR Processing", disable=not progress):
            page_ocr_results = []

            for engine in self.ocr_engines:
                try:
                    result = engine.process_image(image, page_num)
                    page_ocr_results.append(result)

                    # Save intermediate result if configured
                    if self.storage.keep_intermediate:
                        self.storage.save_ocr_result(result, document_name)

                except Exception as e:
                    print(f"Warning: {engine.name} failed on page {page_num}: {e}")

            all_ocr_results.append(page_ocr_results)

        # Step 4: Reconcile results with LLM
        print("Step 4: Reconciling OCR results with LLM...")
        reconciliation_results = []

        for page_num, page_ocr_results in enumerate(
            tqdm(all_ocr_results, desc="Reconciling", disable=not progress),
            start=1
        ):
            if len(page_ocr_results) >= self.reconciler.min_engines_required:
                try:
                    recon_result = self.reconciler.reconcile(
                        page_ocr_results,
                        page_number=page_num
                    )
                    reconciliation_results.append(recon_result)
                except Exception as e:
                    print(f"Warning: Reconciliation failed on page {page_num}: {e}")
                    # Use best single OCR result as fallback
                    best_result = max(page_ocr_results, key=lambda r: r.average_confidence)
                    from .reconciliation.llm_reconciler import ReconciliationResult
                    recon_result = ReconciliationResult(
                        merged_text=best_result.full_text,
                        confidence=best_result.average_confidence,
                        ocr_results=page_ocr_results,
                        metadata={'fallback': True, 'page_number': page_num}
                    )
                    reconciliation_results.append(recon_result)
            else:
                print(f"Warning: Not enough OCR results for page {page_num}, skipping reconciliation")

        # Step 5: Extract metadata
        print("Step 5: Extracting metadata...")
        metadata_config = self.config.get_metadata_config()

        # Combine all reconciled text for metadata extraction
        full_text = '\n\n'.join([r.merged_text for r in reconciliation_results])

        document_metadata = self.metadata_extractor.create_document_metadata(
            pdf_info=pdf_info,
            content_text=full_text,
            extract_dates=metadata_config['extract_dates'],
            extract_authors=metadata_config['extract_authors'],
            extract_recipients=metadata_config['extract_recipients'],
        )

        # Step 6: Save results
        print("Step 6: Saving results...")
        output_path = self.storage.save_document(
            document_name=document_name,
            metadata=document_metadata,
            reconciliation_results=reconciliation_results,
            ocr_results=all_ocr_results if self.storage.keep_intermediate else None
        )

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}\n")

        # Print summary
        print("Summary:")
        print(f"  - Pages processed: {len(page_images)}")
        print(f"  - OCR engines used: {len(self.ocr_engines)}")
        print(f"  - Average confidence: {sum(r.confidence for r in reconciliation_results) / len(reconciliation_results):.2%}")
        if document_metadata.letter_date:
            print(f"  - Date: {document_metadata.letter_date}")
        if document_metadata.letter_author:
            print(f"  - Author: {document_metadata.letter_author}")
        if document_metadata.letter_recipient:
            print(f"  - Recipient: {document_metadata.letter_recipient}")

        total_discrepancies = sum(len(r.discrepancies) for r in reconciliation_results)
        if total_discrepancies > 0:
            print(f"  - Discrepancies flagged: {total_discrepancies}")

        return str(output_path)
