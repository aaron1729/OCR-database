#!/usr/bin/env python3
"""Test pipeline on first few pages."""
from src.config import Config
from src.pipeline import OCRPipeline
import sys

# Initialize
config = Config('config/test_config.yaml')
pipeline = OCRPipeline(config)

# Test on first 2 pages by creating a temporary PDF
# For now, we'll just process the full document but it will show progress

pdf_path = 'data/raw/MAC ARTICLES/Survival of Confederate Revolvers, vol 1, no 2.pdf'

print("Testing pipeline with first pages of Confederate Revolvers article...")
print("Note: This will process all pages but you can Ctrl+C to stop after a few")
print()

try:
    output_path = pipeline.process_document(
        pdf_path=pdf_path,
        document_name='confederate_revolvers_test',
        progress=True
    )

    print(f"\n✓ Processing complete! Results: {output_path}")

    # Show sample results
    full_text = pipeline.storage.get_full_text('confederate_revolvers_test')
    print("\nFirst 1000 characters of merged text:")
    print("="*60)
    print(full_text[:1000])
    print("="*60)

except KeyboardInterrupt:
    print("\n\nStopped by user. Partial results may be saved.")
    sys.exit(0)
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
