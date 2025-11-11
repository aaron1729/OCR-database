"""
PDF processing and conversion to images for OCR.
"""
import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import PyPDF2

# Try to import PDF rendering libraries (fallback order)
try:
    import pypdfium2 as pdfium
    PDF_RENDERER = 'pypdfium2'
except ImportError:
    try:
        from pdf2image import convert_from_path
        PDF_RENDERER = 'pdf2image'
    except ImportError:
        PDF_RENDERER = None


class PDFProcessor:
    """Handles PDF loading and conversion to images."""

    def __init__(self, dpi: int = 300):
        """
        Initialize PDF processor.

        Args:
            dpi: Resolution for PDF to image conversion (higher = better quality)
        """
        self.dpi = dpi

    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with PDF metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            metadata = reader.metadata or {}

            info = {
                'filename': pdf_path.name,
                'num_pages': len(reader.pages),
                'title': metadata.get('/Title', ''),
                'author': metadata.get('/Author', ''),
                'subject': metadata.get('/Subject', ''),
                'creator': metadata.get('/Creator', ''),
                'producer': metadata.get('/Producer', ''),
                'creation_date': metadata.get('/CreationDate', ''),
            }

        return info

    def convert_to_images(
        self,
        pdf_path: str,
        output_dir: str = None
    ) -> List[Tuple[Image.Image, int]]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            output_dir: Optional directory to save images (None = don't save)

        Returns:
            List of tuples (PIL Image, page_number)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if PDF_RENDERER is None:
            raise RuntimeError(
                "No PDF rendering library available. "
                "Install either: pip install pypdfium2  OR  pip install pdf2image (requires poppler)"
            )

        # Convert PDF to images using available renderer
        if PDF_RENDERER == 'pypdfium2':
            images = self._convert_with_pypdfium2(pdf_path)
        else:  # pdf2image
            images = convert_from_path(str(pdf_path), dpi=self.dpi)

        # Create output directory if needed
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Process and optionally save each page
        result = []
        for i, image in enumerate(images, start=1):
            if output_dir:
                output_file = output_path / f"{pdf_path.stem}_page_{i:03d}.png"
                image.save(output_file, 'PNG')

            result.append((image, i))

        return result

    def _convert_with_pypdfium2(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to images using pypdfium2."""
        pdf = pdfium.PdfDocument(str(pdf_path))
        images = []

        # Calculate scale to achieve desired DPI
        # pypdfium2 uses scale factor, 72 DPI is baseline
        scale = self.dpi / 72.0

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            # Render page to PIL Image
            pil_image = page.render(scale=scale).to_pil()
            images.append(pil_image)
            page.close()

        pdf.close()
        return images

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str = None
    ) -> Tuple[dict, List[Tuple[Image.Image, int]]]:
        """
        Complete PDF processing: extract metadata and convert to images.

        Args:
            pdf_path: Path to PDF file
            output_dir: Optional directory to save images

        Returns:
            Tuple of (metadata dict, list of (image, page_number) tuples)
        """
        metadata = self.get_pdf_info(pdf_path)
        images = self.convert_to_images(pdf_path, output_dir)

        return metadata, images
