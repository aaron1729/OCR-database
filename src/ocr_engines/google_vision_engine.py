"""
Google Cloud Vision API engine wrapper.
"""
import os
import io
from PIL import Image
from typing import Optional

try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

from .base import OCREngine, OCRResult, TextBlock, BoundingBox


class GoogleVisionEngine(OCREngine):
    """Wrapper for Google Cloud Vision API."""

    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Google Vision engine.

        Args:
            credentials_path: Path to Google Cloud credentials JSON file
        """
        super().__init__('google_vision')

        # Set credentials path before creating client
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        elif 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
            # Try to load from .env via python-dotenv
            from dotenv import load_dotenv
            load_dotenv()
            # Check for GOOGLE_CLOUD_VISION_CREDENTIALS_PATH and convert to absolute path
            creds_path = os.environ.get('GOOGLE_CLOUD_VISION_CREDENTIALS_PATH')
            if creds_path:
                # Convert relative path to absolute
                abs_path = os.path.abspath(creds_path)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = abs_path

        if GOOGLE_VISION_AVAILABLE:
            try:
                self.client = vision.ImageAnnotatorClient()
            except Exception:
                self.client = None
        else:
            self.client = None

    def is_available(self) -> bool:
        """Check if Google Vision API is available."""
        if not GOOGLE_VISION_AVAILABLE:
            return False

        try:
            # Check if credentials are set
            return 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ
        except Exception:
            return False

    def process_image(self, image: Image.Image, page_number: int = 1) -> OCRResult:
        """
        Process image using Google Cloud Vision.

        Args:
            image: PIL Image to process
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text
        """
        if not self.client:
            raise RuntimeError("Google Vision client not available")

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Create Vision API image
        vision_image = vision.Image(content=img_byte_arr)

        # Perform document text detection (better for handwriting)
        response = self.client.document_text_detection(image=vision_image)

        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        # Extract full text
        full_text = response.full_text_annotation.text if response.full_text_annotation else ""

        # Build text blocks
        text_blocks = []

        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_text = ""
                    block_confidence = 0.0
                    word_count = 0

                    # Get bounding box for block
                    vertices = block.bounding_box.vertices
                    min_x = min(v.x for v in vertices)
                    min_y = min(v.y for v in vertices)
                    max_x = max(v.x for v in vertices)
                    max_y = max(v.y for v in vertices)

                    bbox = BoundingBox(
                        x=float(min_x),
                        y=float(min_y),
                        width=float(max_x - min_x),
                        height=float(max_y - min_y)
                    )

                    # Extract text and confidence from paragraphs/words
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            block_text += word_text + " "
                            block_confidence += word.confidence
                            word_count += 1

                    if block_text.strip():
                        text_blocks.append(TextBlock(
                            text=block_text.strip(),
                            confidence=block_confidence / word_count if word_count > 0 else 0.0,
                            bounding_box=bbox
                        ))

        return OCRResult(
            engine_name=self.name,
            full_text=full_text.strip(),
            text_blocks=text_blocks,
            page_number=page_number,
            metadata={
                'api': 'google_cloud_vision',
                'detection_type': 'document_text_detection'
            }
        )
