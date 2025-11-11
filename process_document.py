#!/usr/bin/env python3
"""
Command-line interface for processing historical documents with OCR.
"""
import argparse
import sys
from pathlib import Path

from src.config import Config
from src.pipeline import OCRPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Process historical handwritten documents with multi-OCR and LLM reconciliation'
    )

    parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to PDF file to process'
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/example_config.yaml',
        help='Path to configuration file (default: config/example_config.yaml)'
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        help='Document name (default: use PDF filename)'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    args = parser.parse_args()

    # Check if PDF exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        print("Using default configuration...", file=sys.stderr)
        config = Config()

    # Initialize pipeline
    try:
        pipeline = OCRPipeline(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    # Process document
    try:
        output_path = pipeline.process_document(
            pdf_path=str(pdf_path),
            document_name=args.name,
            progress=not args.no_progress
        )
        print(f"\nSuccess! Results saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"\nError processing document: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
