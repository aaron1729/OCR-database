#!/usr/bin/env python3
"""Test pipeline on first 10 pages of Atlanta letters (1864 handwritten)."""
from src.config import Config
from src.preprocessing import PDFProcessor, ImageEnhancer
from src.ocr_engines import TesseractEngine
from src.reconciliation import LLMReconciler
from src.reconciliation.llm_providers import AnthropicProvider
from src.metadata import MetadataExtractor
from src.storage import JSONStorage
from tqdm import tqdm

# Initialize components
print("Initializing pipeline for handwritten letters from 1864...")
config = Config('config/test_config.yaml')

pdf_processor = PDFProcessor(dpi=300)
image_enhancer = ImageEnhancer(**config.get_preprocessing_config())
tesseract = TesseractEngine()

# LLM reconciliation
llm_config = config.get_llm_config()
recon_config = config.get_reconciliation_config()
provider = AnthropicProvider(
    model=llm_config['model'],
    temperature=llm_config['temperature'],
    max_tokens=llm_config['max_tokens']
)
reconciler = LLMReconciler(
    llm_provider=provider,
    flag_threshold=recon_config['flag_threshold'],
    min_engines_required=recon_config['min_engines_required']
)

metadata_extractor = MetadataExtractor()
storage = JSONStorage('data/output', keep_intermediate=True)

# Process PDF
pdf_path = 'data/raw/NARA Ordnance/109-4-16, letters sent 1864 Atlanta.pdf'
document_name = 'atlanta_letters_1864_first10'

print(f"\nProcessing: {pdf_path}")
print("Note: Processing first 10 pages only\n")
print("="*60)

# Step 1: Get PDF info and convert first 10 pages
print("Step 1: Converting first 10 pages to images...")
pdf_info = pdf_processor.get_pdf_info(pdf_path)
all_images = pdf_processor.convert_to_images(pdf_path)
page_images = all_images[:10]  # First 10 pages only
print(f"Converted {len(page_images)} pages")

# Step 2: Enhance images
print("Step 2: Enhancing images...")
enhanced_images = []
for image, page_num in tqdm(page_images, desc="Enhancing"):
    enhanced = image_enhancer.enhance(image)
    enhanced_images.append((enhanced, page_num))

# Step 3: OCR
print("Step 3: Running Tesseract OCR...")
ocr_results = []
for image, page_num in tqdm(enhanced_images, desc="OCR Processing"):
    result = tesseract.process_image(image, page_num)
    ocr_results.append([result])  # List of lists for reconciliation
    if storage.keep_intermediate:
        storage.save_ocr_result(result, document_name)

# Step 4: Reconciliation
print("Step 4: Reconciling with Claude...")
reconciliation_results = []
for page_num, page_ocr_results in enumerate(tqdm(ocr_results, desc="Reconciling"), start=1):
    recon_result = reconciler.reconcile(page_ocr_results, page_number=page_num)
    reconciliation_results.append(recon_result)

# Step 5: Metadata
print("Step 5: Extracting metadata...")
full_text = '\n\n'.join([r.merged_text for r in reconciliation_results])
metadata_config = config.get_metadata_config()
document_metadata = metadata_extractor.create_document_metadata(
    pdf_info=pdf_info,
    content_text=full_text,
    extract_dates=metadata_config['extract_dates'],
    extract_authors=metadata_config['extract_authors'],
    extract_recipients=metadata_config['extract_recipients'],
)

# Step 6: Save
print("Step 6: Saving results...")
output_path = storage.save_document(
    document_name=document_name,
    metadata=document_metadata,
    reconciliation_results=reconciliation_results,
    ocr_results=ocr_results
)

print("\n" + "="*60)
print("Processing complete!")
print("="*60)
print(f"Output: {output_path}")
print(f"\nSummary:")
print(f"  - Pages processed: {len(page_images)}")
print(f"  - Average confidence: {sum(r.confidence for r in reconciliation_results) / len(reconciliation_results):.2%}")
if document_metadata.letter_date:
    print(f"  - Date found: {document_metadata.letter_date}")
if document_metadata.letter_author:
    print(f"  - Author: {document_metadata.letter_author}")
if document_metadata.letter_recipient:
    print(f"  - Recipient: {document_metadata.letter_recipient}")

total_discrepancies = sum(len(r.discrepancies) for r in reconciliation_results)
print(f"  - Discrepancies flagged: {total_discrepancies}")

# Show first page sample
print("\n" + "="*60)
print("FIRST PAGE - Sample Text (first 800 chars):")
print("="*60)
print(reconciliation_results[0].merged_text[:800])
print("="*60)

print("\n✓ Test complete!")
