"""
Base classes for OCR engines.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from PIL import Image


@dataclass
class BoundingBox:
    """Represents a bounding box for text."""
    x: float
    y: float
    width: float
    height: float


@dataclass
class TextBlock:
    """Represents a block of text with metadata."""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    language: Optional[str] = None


@dataclass
class OCRResult:
    """Result from an OCR engine."""
    engine_name: str
    full_text: str
    text_blocks: List[TextBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    average_confidence: float = 0.0
    page_number: Optional[int] = None

    def __post_init__(self):
        """Calculate average confidence if not provided."""
        if self.average_confidence == 0.0 and self.text_blocks:
            confidences = [block.confidence for block in self.text_blocks]
            self.average_confidence = sum(confidences) / len(confidences)


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    def __init__(self, name: str):
        """
        Initialize OCR engine.

        Args:
            name: Name of the OCR engine
        """
        self.name = name

    @abstractmethod
    def process_image(self, image: Image.Image, page_number: int = 1) -> OCRResult:
        """
        Process an image and extract text.

        Args:
            image: PIL Image to process
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text and metadata
        """
        pass

    def is_available(self) -> bool:
        """
        Check if the OCR engine is available and properly configured.

        Returns:
            True if engine can be used, False otherwise
        """
        try:
            # Subclasses should override to check API keys, installations, etc.
            return True
        except Exception:
            return False
