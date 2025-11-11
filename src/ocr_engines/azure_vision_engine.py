"""
Azure Computer Vision API engine wrapper.
"""
import io
import os
import time
from PIL import Image
from typing import Optional

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
    from msrest.authentication import CognitiveServicesCredentials
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from .base import OCREngine, OCRResult, TextBlock, BoundingBox


class AzureVisionEngine(OCREngine):
    """Wrapper for Azure Computer Vision API."""

    def __init__(
        self,
        subscription_key: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        """
        Initialize Azure Vision engine.

        Args:
            subscription_key: Azure subscription key (or use environment variable)
            endpoint: Azure endpoint URL (or use environment variable)
        """
        super().__init__('azure_vision')

        if not AZURE_AVAILABLE:
            self.client = None
            return

        key = subscription_key or os.environ.get('AZURE_COMPUTER_VISION_KEY')
        ep = endpoint or os.environ.get('AZURE_COMPUTER_VISION_ENDPOINT')

        if key and ep:
            credentials = CognitiveServicesCredentials(key)
            self.client = ComputerVisionClient(ep, credentials)
        else:
            self.client = None

    def is_available(self) -> bool:
        """Check if Azure Vision API is available."""
        return AZURE_AVAILABLE and self.client is not None

    def process_image(self, image: Image.Image, page_number: int = 1) -> OCRResult:
        """
        Process image using Azure Computer Vision.

        Args:
            image: PIL Image to process
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text
        """
        if not self.client:
            raise RuntimeError("Azure Vision client not available")

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Call Azure Read API (best for handwriting)
        read_operation = self.client.read_in_stream(img_byte_arr, raw=True)

        # Get operation location (URL with operation ID)
        operation_location = read_operation.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        # Wait for the operation to complete
        max_wait = 30  # seconds
        wait_interval = 1
        elapsed = 0

        while elapsed < max_wait:
            result = self.client.get_read_result(operation_id)

            if result.status == OperationStatusCodes.succeeded:
                break

            if result.status == OperationStatusCodes.failed:
                raise Exception("Azure Read operation failed")

            time.sleep(wait_interval)
            elapsed += wait_interval

        if result.status != OperationStatusCodes.succeeded:
            raise Exception("Azure Read operation timed out")

        # Extract text and blocks
        full_text_parts = []
        text_blocks = []

        for read_result in result.analyze_result.read_results:
            for line in read_result.lines:
                text = line.text
                # Azure doesn't provide confidence at line level, use word-level average
                word_confidences = [word.confidence for word in line.words if hasattr(word, 'confidence')]
                confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0.5

                # Get bounding box (Azure provides polygon points)
                bbox_points = line.bounding_box
                min_x = min(bbox_points[0::2])  # Even indices are x coordinates
                max_x = max(bbox_points[0::2])
                min_y = min(bbox_points[1::2])  # Odd indices are y coordinates
                max_y = max(bbox_points[1::2])

                bbox = BoundingBox(
                    x=float(min_x),
                    y=float(min_y),
                    width=float(max_x - min_x),
                    height=float(max_y - min_y)
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
                'api': 'azure_computer_vision',
                'detection_type': 'read_api'
            }
        )
