"""
AWS Textract engine wrapper.
"""
import io
import os
from PIL import Image
from typing import Optional

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

from .base import OCREngine, OCRResult, TextBlock, BoundingBox


class AWSTextractEngine(OCREngine):
    """Wrapper for AWS Textract."""

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = 'us-east-1'
    ):
        """
        Initialize AWS Textract engine.

        Args:
            aws_access_key_id: AWS access key (or use environment variable)
            aws_secret_access_key: AWS secret key (or use environment variable)
            region_name: AWS region
        """
        super().__init__('aws_textract')

        if not AWS_AVAILABLE:
            self.client = None
            return

        # Initialize boto3 client
        if aws_access_key_id and aws_secret_access_key:
            self.client = boto3.client(
                'textract',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            # Use environment variables or IAM role
            self.client = boto3.client('textract', region_name=region_name)

    def is_available(self) -> bool:
        """Check if AWS Textract is available."""
        if not AWS_AVAILABLE or not self.client:
            return False

        try:
            # Try to use the client
            self.client.meta.region_name
            return True
        except Exception:
            return False

    def process_image(self, image: Image.Image, page_number: int = 1) -> OCRResult:
        """
        Process image using AWS Textract.

        Args:
            image: PIL Image to process
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text
        """
        if not self.client:
            raise RuntimeError("AWS Textract client not available")

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Call Textract
        response = self.client.detect_document_text(
            Document={'Bytes': img_byte_arr.read()}
        )

        # Extract text and blocks
        full_text_parts = []
        text_blocks = []

        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                text = block['Text']
                confidence = block['Confidence'] / 100.0  # Convert to 0-1 range

                # Get bounding box
                bbox_data = block['Geometry']['BoundingBox']
                # Textract returns normalized coordinates (0-1)
                # Convert to pixel coordinates (approximate based on image size)
                img_width, img_height = image.size
                bbox = BoundingBox(
                    x=bbox_data['Left'] * img_width,
                    y=bbox_data['Top'] * img_height,
                    width=bbox_data['Width'] * img_width,
                    height=bbox_data['Height'] * img_height
                )

                text_blocks.append(TextBlock(
                    text=text,
                    confidence=confidence,
                    bounding_box=bbox
                ))

                full_text_parts.append(text)

        full_text = '\n'.join(full_text_parts)

        return OCRResult(
            engine_name=self.name,
            full_text=full_text,
            text_blocks=text_blocks,
            page_number=page_number,
            metadata={
                'api': 'aws_textract',
                'detection_type': 'detect_document_text'
            }
        )
