#!/usr/bin/env python3
"""Quick test on a single page."""
from src.preprocessing import PDFProcessor, ImageEnhancer
from src.ocr_engines import TesseractEngine
from src.config import Config

# Initialize components
print("Initializing components...")
config = Config('config/test_config.yaml')
pdf_processor = PDFProcessor(dpi=200)  # Lower DPI for speed
image_enhancer = ImageEnhancer(**config.get_preprocessing_config())
tesseract = TesseractEngine()

# Check Tesseract
print(f"Tesseract available: {tesseract.is_available()}")

# Process PDF
pdf_path = 'data/raw/MAC ARTICLES/Survival of Confederate Revolvers, vol 1, no 2.pdf'
print(f"\nProcessing: {pdf_path}")

# Get first page
print("Converting first page...")
pdf_info, page_images = pdf_processor.process_pdf(pdf_path)
first_image, page_num = page_images[0]

print(f"Page size: {first_image.size}")

# Enhance image
print("Enhancing image...")
enhanced = image_enhancer.enhance(first_image)

# Run OCR
print("Running Tesseract OCR...")
result = tesseract.process_image(enhanced, page_num)

# Display results
print("\n" + "="*60)
print(f"OCR Results (Page {page_num}):")
print("="*60)
print(f"Confidence: {result.average_confidence:.2%}")
print(f"Text blocks: {len(result.text_blocks)}")
print("\nFirst 500 characters of text:")
print("-"*60)
print(result.full_text[:500])
print("-"*60)

print("\n✓ Single page test completed successfully!")
