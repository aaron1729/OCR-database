"""
Metadata extraction from historical letters.
"""
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class DocumentMetadata:
    """Metadata for a historical document."""
    # Core document info
    filename: str
    num_pages: int

    # PDF metadata
    pdf_title: Optional[str] = None
    pdf_author: Optional[str] = None
    pdf_subject: Optional[str] = None
    creation_date: Optional[str] = None

    # Extracted from content
    letter_date: Optional[str] = None
    letter_author: Optional[str] = None
    letter_recipient: Optional[str] = None
    location: Optional[str] = None

    # Additional fields
    language: str = "en"
    document_type: str = "letter"
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


class MetadataExtractor:
    """Extract metadata from historical letters."""

    def __init__(self):
        """Initialize metadata extractor."""
        # Common date patterns in historical letters
        self.date_patterns = [
            # "January 15, 1850"
            r'([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
            # "15th January 1850"
            r'(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4})',
            # "Jan. 15, 1850"
            r'([A-Z][a-z]{2}\.\s+\d{1,2},?\s+\d{4})',
            # "1850-01-15" or "15/01/1850"
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
        ]

        # Patterns for salutations and signatures
        self.salutation_patterns = [
            r'Dear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'My\s+dear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'To\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]

        self.signature_patterns = [
            r'Yours\s+(?:truly|sincerely|faithfully),?\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Sincerely,?\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'With\s+(?:love|regards),?\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]

        # Location patterns
        self.location_patterns = [
            # "Written from London" or "From Boston"
            r'(?:Written\s+from|From)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # City, State pattern
            r'([A-Z][a-z]+,\s*[A-Z]{2})',
        ]

    def extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text.

        Args:
            text: Text to extract dates from

        Returns:
            List of found dates
        """
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return dates

    def extract_recipient(self, text: str) -> Optional[str]:
        """
        Extract recipient from letter salutation.

        Args:
            text: Text to extract recipient from

        Returns:
            Recipient name if found
        """
        # Check first few lines for salutation
        lines = text.split('\n')[:10]
        first_part = '\n'.join(lines)

        for pattern in self.salutation_patterns:
            match = re.search(pattern, first_part)
            if match:
                return match.group(1).strip()

        return None

    def extract_author(self, text: str) -> Optional[str]:
        """
        Extract author from letter signature.

        Args:
            text: Text to extract author from

        Returns:
            Author name if found
        """
        # Check last few lines for signature
        lines = text.split('\n')
        last_part = '\n'.join(lines[-15:])

        for pattern in self.signature_patterns:
            match = re.search(pattern, last_part, re.MULTILINE)
            if match:
                return match.group(1).strip()

        return None

    def extract_location(self, text: str) -> Optional[str]:
        """
        Extract location from letter.

        Args:
            text: Text to extract location from

        Returns:
            Location if found
        """
        # Check first few lines
        lines = text.split('\n')[:10]
        first_part = '\n'.join(lines)

        for pattern in self.location_patterns:
            match = re.search(pattern, first_part)
            if match:
                return match.group(1).strip()

        return None

    def extract_from_text(
        self,
        text: str,
        extract_dates: bool = True,
        extract_authors: bool = True,
        extract_recipients: bool = True,
        extract_locations: bool = True
    ) -> Dict[str, Any]:
        """
        Extract all metadata from text.

        Args:
            text: Text to extract from
            extract_dates: Whether to extract dates
            extract_authors: Whether to extract authors
            extract_recipients: Whether to extract recipients
            extract_locations: Whether to extract locations

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}

        if extract_dates:
            dates = self.extract_dates(text)
            if dates:
                metadata['letter_date'] = dates[0]  # Use first found date

        if extract_recipients:
            recipient = self.extract_recipient(text)
            if recipient:
                metadata['letter_recipient'] = recipient

        if extract_authors:
            author = self.extract_author(text)
            if author:
                metadata['letter_author'] = author

        if extract_locations:
            location = self.extract_location(text)
            if location:
                metadata['location'] = location

        return metadata

    def create_document_metadata(
        self,
        pdf_info: Dict[str, Any],
        content_text: str,
        extract_dates: bool = True,
        extract_authors: bool = True,
        extract_recipients: bool = True,
        extract_locations: bool = True
    ) -> DocumentMetadata:
        """
        Create complete document metadata.

        Args:
            pdf_info: PDF metadata from PDFProcessor
            content_text: Extracted text content
            extract_dates: Whether to extract dates
            extract_authors: Whether to extract authors
            extract_recipients: Whether to extract recipients
            extract_locations: Whether to extract locations

        Returns:
            DocumentMetadata object
        """
        # Extract from content
        content_metadata = self.extract_from_text(
            content_text,
            extract_dates,
            extract_authors,
            extract_recipients,
            extract_locations
        )

        # Combine with PDF metadata
        return DocumentMetadata(
            filename=pdf_info['filename'],
            num_pages=pdf_info['num_pages'],
            pdf_title=pdf_info.get('title'),
            pdf_author=pdf_info.get('author'),
            pdf_subject=pdf_info.get('subject'),
            creation_date=pdf_info.get('creation_date'),
            letter_date=content_metadata.get('letter_date'),
            letter_author=content_metadata.get('letter_author'),
            letter_recipient=content_metadata.get('letter_recipient'),
            location=content_metadata.get('location'),
        )
