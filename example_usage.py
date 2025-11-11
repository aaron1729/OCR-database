#!/usr/bin/env python3
"""
Example usage of the OCR pipeline programmatically.
"""
from src.config import Config
from src.pipeline import OCRPipeline


def main():
    """Example: Process a document programmatically."""

    # Option 1: Use configuration file
    config = Config('config/example_config.yaml')

    # Option 2: Create config programmatically
    # config = Config()  # Uses defaults and environment variables

    # Initialize pipeline
    pipeline = OCRPipeline(config)

    # Process a document
    pdf_path = 'data/input/sample_letter.pdf'  # Replace with your PDF path
    document_name = 'sample_letter_1850'  # Optional custom name

    print("Starting OCR processing...")
    output_path = pipeline.process_document(
        pdf_path=pdf_path,
        document_name=document_name,
        progress=True
    )

    print(f"\nProcessing complete! Output saved to: {output_path}")

    # Retrieve the full text
    full_text = pipeline.storage.get_full_text(document_name)
    print("\n" + "="*60)
    print("MERGED TEXT:")
    print("="*60)
    print(full_text)

    # Load complete results
    results = pipeline.storage.load_document(document_name)

    # Access metadata
    metadata = results['metadata']
    print("\n" + "="*60)
    print("METADATA:")
    print("="*60)
    print(f"Date: {metadata.get('letter_date', 'N/A')}")
    print(f"Author: {metadata.get('letter_author', 'N/A')}")
    print(f"Recipient: {metadata.get('letter_recipient', 'N/A')}")
    print(f"Location: {metadata.get('location', 'N/A')}")

    # Check for discrepancies
    print("\n" + "="*60)
    print("DISCREPANCIES:")
    print("="*60)
    for page in results['pages']:
        page_num = page['page_number']
        discrepancies = page['reconciliation']['discrepancies']

        if discrepancies:
            print(f"\nPage {page_num}:")
            for i, disc in enumerate(discrepancies, 1):
                print(f"  {i}. {disc['position']}")
                print(f"     Variants: {disc['variants']}")
                print(f"     Context: {disc['context']}")
        else:
            print(f"\nPage {page_num}: No discrepancies")


if __name__ == '__main__':
    main()
