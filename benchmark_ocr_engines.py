#!/usr/bin/env python3
"""
Benchmark multiple OCR engines on challenging handwritten text.
Tests all 20 pages of 1864 Atlanta letters.
"""
import json
import time
from pathlib import Path
from src.preprocessing import PDFProcessor, ImageEnhancer
from src.ocr_engines import TesseractEngine, GoogleVisionEngine, AWSTextractEngine, AzureVisionEngine
from PIL import Image
import numpy as np

# Also test PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

print("="*70)
print("OCR ENGINE BENCHMARK - 1864 Handwritten Letters")
print("="*70)
print()

# Configuration
pdf_path = 'data/raw/ZZZ samples for OCR tests/109-4-31, 4-63--8-64 Macon Armory letters -- PAGES 1-20 ONLY.pdf'
output_dir = Path('benchmark_results')
output_dir.mkdir(exist_ok=True)

# Process PDF at different DPIs for Tesseract comparison
print("Step 1: Preparing test images...")
processor_300 = PDFProcessor(dpi=300)
processor_150 = PDFProcessor(dpi=150)

pdf_info = processor_300.get_pdf_info(pdf_path)
print(f"Document: {pdf_info['filename']}")
print(f"Testing all {pdf_info['num_pages']} pages\n")

# Get images at both DPIs
images_300 = processor_300.convert_to_images(pdf_path)
images_150 = processor_150.convert_to_images(pdf_path)

# Enhance images
enhancer = ImageEnhancer(deskew=True, enhance_contrast=True, denoise=True, binarize=False)
enhanced_300 = [(enhancer.enhance(img), page) for img, page in images_300]
enhanced_150 = [(enhancer.enhance(img), page) for img, page in images_150]

# Also try binarized version
enhancer_bin = ImageEnhancer(deskew=True, enhance_contrast=True, denoise=True, binarize=True)
binarized_300 = [(enhancer_bin.enhance(img), page) for img, page in images_300]

print(f"✓ Prepared images at 300 DPI, 150 DPI, and binarized\n")

# Initialize engines
print("Step 2: Initializing OCR engines...")
engines = []

# Tesseract variants
tesseract_300 = TesseractEngine()
engines.append(("Tesseract (300 DPI, enhanced)", tesseract_300, enhanced_300))

tesseract_150 = TesseractEngine()
engines.append(("Tesseract (150 DPI, enhanced)", tesseract_150, enhanced_150))

tesseract_bin = TesseractEngine()
engines.append(("Tesseract (300 DPI, binarized)", tesseract_bin, binarized_300))

# PaddleOCR
if PADDLE_AVAILABLE:
    print("  - PaddleOCR available")
    paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    engines.append(("PaddleOCR (handwriting)", paddle_ocr, enhanced_300))
else:
    print("  - PaddleOCR not available")

# Cloud APIs
google_engine = GoogleVisionEngine()
if google_engine.is_available():
    print("  - Google Vision API available")
    engines.append(("Google Vision API", google_engine, enhanced_300))

aws_engine = AWSTextractEngine()
if aws_engine.is_available():
    print("  - AWS Textract available")
    engines.append(("AWS Textract", aws_engine, enhanced_300))

azure_engine = AzureVisionEngine()
if azure_engine.is_available():
    print("  - Azure Vision API available")
    engines.append(("Azure Vision API", azure_engine, enhanced_300))

print(f"\n✓ {len(engines)} engine configurations ready\n")

# Run benchmark
results = {}
print("Step 3: Running OCR benchmark...")
print("="*70)

for engine_name, engine, images in engines:
    print(f"\nTesting: {engine_name}")
    print("-"*70)

    engine_results = []
    total_time = 0

    for img, page_num in images:
        start_time = time.time()

        try:
            # Handle PaddleOCR differently
            if isinstance(engine, type(paddle_ocr)) if PADDLE_AVAILABLE else False:
                # Convert PIL to numpy for PaddleOCR
                img_array = np.array(img)
                paddle_result = engine.ocr(img_array, cls=True)

                # Extract text and confidence
                if paddle_result and paddle_result[0]:
                    texts = []
                    confidences = []
                    for line in paddle_result[0]:
                        if line[1][0]:  # text
                            texts.append(line[1][0])
                            confidences.append(line[1][1])  # confidence

                    full_text = '\n'.join(texts)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                else:
                    full_text = ""
                    avg_confidence = 0.0

                result = {
                    'full_text': full_text,
                    'average_confidence': avg_confidence,
                    'page_number': page_num
                }
            else:
                # Standard OCR engines
                ocr_result = engine.process_image(img, page_num)
                result = {
                    'full_text': ocr_result.full_text,
                    'average_confidence': ocr_result.average_confidence,
                    'page_number': page_num
                }

            elapsed = time.time() - start_time
            total_time += elapsed

            engine_results.append(result)
            print(f"  Page {page_num}: {elapsed:.2f}s, confidence: {result['average_confidence']:.1%}")

        except Exception as e:
            print(f"  Page {page_num}: ERROR - {str(e)}")
            engine_results.append({
                'full_text': '',
                'average_confidence': 0.0,
                'page_number': page_num,
                'error': str(e)
            })

    # Calculate averages
    avg_conf = sum(r['average_confidence'] for r in engine_results) / len(engine_results)
    avg_time = total_time / len(engine_results)

    results[engine_name] = {
        'results': engine_results,
        'avg_confidence': avg_conf,
        'avg_time_per_page': avg_time,
        'total_time': total_time
    }

    print(f"  Average confidence: {avg_conf:.1%}")
    print(f"  Average time/page: {avg_time:.2f}s")

print("\n" + "="*70)
print("BENCHMARK COMPLETE")
print("="*70)

# Save results
output_file = output_dir / 'benchmark_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")

# Print summary
print("\n" + "="*70)
print("SUMMARY - Ranked by Confidence")
print("="*70)

ranked = sorted(results.items(), key=lambda x: x[1]['avg_confidence'], reverse=True)

print(f"\n{'Rank':<6} {'Engine':<40} {'Confidence':<12} {'Time/page'}")
print("-"*70)

for i, (engine_name, data) in enumerate(ranked, 1):
    conf = data['avg_confidence']
    time_per_page = data['avg_time_per_page']
    print(f"{i:<6} {engine_name:<40} {conf:>10.1%}  {time_per_page:>7.2f}s")

# Show sample text from best engine
print("\n" + "="*70)
print(f"SAMPLE TEXT - Best Engine: {ranked[0][0]}")
print("="*70)
best_text = ranked[0][1]['results'][0]['full_text']
print(best_text[:1000] if best_text else "[No text extracted]")
print("="*70)

print("\n✓ Benchmark complete! Check benchmark_results/ for detailed outputs.")
