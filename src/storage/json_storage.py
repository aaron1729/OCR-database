"""
JSON-based storage for OCR results and metadata.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict
from datetime import datetime

from ..ocr_engines.base import OCRResult
from ..reconciliation.llm_reconciler import ReconciliationResult
from ..metadata.extractor import DocumentMetadata


class JSONStorage:
    """Store OCR results and metadata as JSON files."""

    def __init__(self, output_dir: str, keep_intermediate: bool = True):
        """
        Initialize JSON storage.

        Args:
            output_dir: Directory to save JSON files
            keep_intermediate: Whether to keep individual OCR results
        """
        self.output_dir = Path(output_dir)
        self.keep_intermediate = keep_intermediate
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        if self.keep_intermediate:
            (self.output_dir / 'ocr_results').mkdir(exist_ok=True)

    def _serialize_ocr_result(self, result: OCRResult) -> Dict[str, Any]:
        """Convert OCRResult to JSON-serializable dict."""
        return {
            'engine_name': result.engine_name,
            'full_text': result.full_text,
            'average_confidence': result.average_confidence,
            'page_number': result.page_number,
            'text_blocks': [
                {
                    'text': block.text,
                    'confidence': block.confidence,
                    'bounding_box': {
                        'x': block.bounding_box.x,
                        'y': block.bounding_box.y,
                        'width': block.bounding_box.width,
                        'height': block.bounding_box.height
                    } if block.bounding_box else None,
                    'language': block.language
                }
                for block in result.text_blocks
            ],
            'metadata': result.metadata
        }

    def _serialize_discrepancy(self, discrepancy) -> Dict[str, Any]:
        """Convert Discrepancy to JSON-serializable dict."""
        return {
            'position': discrepancy.position,
            'variants': discrepancy.variants,
            'engines': discrepancy.engines,
            'similarity': discrepancy.similarity,
            'context': discrepancy.context
        }

    def _serialize_reconciliation_result(
        self,
        result: ReconciliationResult
    ) -> Dict[str, Any]:
        """Convert ReconciliationResult to JSON-serializable dict."""
        return {
            'merged_text': result.merged_text,
            'confidence': result.confidence,
            'discrepancies': [
                self._serialize_discrepancy(d) for d in result.discrepancies
            ],
            'metadata': result.metadata
        }

    def save_ocr_result(
        self,
        result: OCRResult,
        document_name: str
    ) -> Path:
        """
        Save individual OCR result.

        Args:
            result: OCR result to save
            document_name: Base name for the document

        Returns:
            Path to saved file
        """
        if not self.keep_intermediate:
            return None

        filename = f"{document_name}_page{result.page_number}_{result.engine_name}.json"
        filepath = self.output_dir / 'ocr_results' / filename

        data = self._serialize_ocr_result(result)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def save_document(
        self,
        document_name: str,
        metadata: DocumentMetadata,
        reconciliation_results: List[ReconciliationResult],
        ocr_results: List[List[OCRResult]] = None
    ) -> Path:
        """
        Save complete document with all results.

        Args:
            document_name: Base name for the document
            metadata: Document metadata
            reconciliation_results: List of reconciliation results (one per page)
            ocr_results: Optional list of OCR results per page

        Returns:
            Path to saved file
        """
        filename = f"{document_name}.json"
        filepath = self.output_dir / filename

        # Build complete document data
        data = {
            'document_name': document_name,
            'processing_date': datetime.now().isoformat(),
            'metadata': asdict(metadata),
            'pages': []
        }

        # Add reconciliation results
        for i, recon_result in enumerate(reconciliation_results, 1):
            page_data = {
                'page_number': i,
                'reconciliation': self._serialize_reconciliation_result(recon_result)
            }

            # Add individual OCR results if available
            if ocr_results and i <= len(ocr_results):
                page_data['ocr_results'] = [
                    self._serialize_ocr_result(ocr_result)
                    for ocr_result in ocr_results[i - 1]
                ]

            data['pages'].append(page_data)

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def load_document(self, document_name: str) -> Dict[str, Any]:
        """
        Load a saved document.

        Args:
            document_name: Name of the document to load

        Returns:
            Document data as dictionary
        """
        filename = f"{document_name}.json"
        filepath = self.output_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_documents(self) -> List[str]:
        """
        List all saved documents.

        Returns:
            List of document names
        """
        json_files = self.output_dir.glob('*.json')
        return [f.stem for f in json_files]

    def get_full_text(self, document_name: str) -> str:
        """
        Get the complete merged text from a document.

        Args:
            document_name: Name of the document

        Returns:
            Full merged text
        """
        data = self.load_document(document_name)

        text_parts = []
        for page in data['pages']:
            merged_text = page['reconciliation']['merged_text']
            text_parts.append(merged_text)

        return '\n\n'.join(text_parts)
