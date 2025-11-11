"""
Image enhancement for improved OCR accuracy on historical handwritten documents.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Union
from skimage import restoration
from skimage.filters import threshold_sauvola


class ImageEnhancer:
    """Enhances images for better OCR results on handwritten historical documents."""

    def __init__(
        self,
        deskew: bool = True,
        enhance_contrast: bool = True,
        denoise: bool = True,
        binarize: bool = False
    ):
        """
        Initialize image enhancer.

        Args:
            deskew: Correct rotation/skew
            enhance_contrast: Improve contrast
            denoise: Remove noise
            binarize: Convert to binary (black/white) image
        """
        self.deskew = deskew
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
        self.binarize = binarize

    def pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format."""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew in the image.

        Args:
            image: Input image (BGR format)

        Returns:
            Deskewed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return image

        # Calculate angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)

        # Get median angle
        median_angle = np.median(angles)

        # Only correct if angle is significant (> 0.5 degrees)
        if abs(median_angle) > 0.5:
            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated

        return image

    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input image (BGR format)

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge channels
        enhanced_lab = cv2.merge([l_enhanced, a, b])

        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise while preserving text edges.

        Args:
            image: Input image (BGR format)

        Returns:
            Denoised image
        """
        # Use bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # Additional non-local means denoising for better results
        denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)

        return denoised

    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to binary image using adaptive thresholding.

        Args:
            image: Input image (BGR format)

        Returns:
            Binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use Sauvola thresholding (good for historical documents)
        # This is better than Otsu for documents with uneven illumination
        thresh_sauvola = threshold_sauvola(gray, window_size=25)
        binary = (gray > thresh_sauvola).astype(np.uint8) * 255

        # Convert back to BGR for consistency
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        return binary_bgr

    def enhance(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply all enabled enhancement operations.

        Args:
            image: Input image (PIL Image or numpy array)

        Returns:
            Enhanced PIL Image
        """
        # Convert to OpenCV format if needed
        if isinstance(image, Image.Image):
            img = self.pil_to_cv2(image)
        else:
            img = image.copy()

        # Apply enhancements in order
        if self.deskew:
            img = self.correct_skew(img)

        if self.enhance_contrast:
            img = self.enhance_contrast_clahe(img)

        if self.denoise:
            img = self.remove_noise(img)

        if self.binarize:
            img = self.binarize_image(img)

        # Convert back to PIL
        return self.cv2_to_pil(img)
