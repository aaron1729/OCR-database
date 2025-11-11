"""
Tesseract OCR engine wrapper.
"""
import pytesseract
from PIL import Image
from typing import Optional
import json

from .base import OCREngine, OCRResult, TextBlock, BoundingBox


class TesseractEngine(OCREngine):
    """Wrapper for Tesseract OCR."""

    def __init__(self, language: str = 'eng', config: Optional[str] = None):
        """
        Initialize Tesseract engine.

        Args:
            language: Language code(s) for OCR (e.g., 'eng', 'eng+fra')
            config: Custom Tesseract config string
        """
        super().__init__('tesseract')
        self.language = language
        # Config optimized for handwriting
        self.config = config or '--psm 6 --oem 3'

    def is_available(self) -> bool:
        """Check if Tesseract is installed and available."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def process_image(self, image: Image.Image, page_number: int = 1) -> OCRResult:
        """
        Process image using Tesseract.

        Args:
            image: PIL Image to process
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text
        """
        # Get detailed data from Tesseract
        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=self.config,
            output_type=pytesseract.Output.DICT
        )

        # Extract full text
        full_text = pytesseract.image_to_string(
            image,
            lang=self.language,
            config=self.config
        )

        # Build text blocks from detailed data
        text_blocks = []
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:  # Skip empty text
                continue

            confidence = float(data['conf'][i])
            if confidence < 0:  # Tesseract returns -1 for no confidence
                confidence = 0.0
            else:
                confidence = confidence / 100.0  # Convert to 0-1 range

            bbox = BoundingBox(
                x=float(data['left'][i]),
                y=float(data['top'][i]),
                width=float(data['width'][i]),
                height=float(data['height'][i])
            )

            text_blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bounding_box=bbox
            ))

        return OCRResult(
            engine_name=self.name,
            full_text=full_text.strip(),
            text_blocks=text_blocks,
            page_number=page_number,
            metadata={
                'language': self.language,
                'config': self.config
            }
        )
