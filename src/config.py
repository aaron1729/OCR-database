"""
Configuration management for OCR pipeline.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager for OCR pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file (optional)
        """
        # Load environment variables
        load_dotenv()

        # Load YAML config if provided
        self.config_data = {}
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}

    def get_ocr_engines(self) -> list:
        """Get list of enabled OCR engines."""
        return self.config_data.get('ocr', {}).get('engines', ['tesseract'])

    def get_preprocessing_config(self) -> Dict[str, bool]:
        """Get preprocessing configuration."""
        default = {
            'deskew': True,
            'enhance_contrast': True,
            'denoise': True,
            'binarize': False
        }
        return self.config_data.get('ocr', {}).get('preprocessing', default)

    def get_confidence_threshold(self) -> float:
        """Get OCR confidence threshold."""
        return self.config_data.get('ocr', {}).get('confidence_threshold', 0.7)

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        default = {
            'provider': os.environ.get('LLM_PROVIDER', 'openai'),
            'model': os.environ.get('LLM_MODEL', 'gpt-4o'),
            'temperature': 0.2,
            'max_tokens': 4000
        }
        return self.config_data.get('llm', default)

    def get_reconciliation_config(self) -> Dict[str, Any]:
        """Get reconciliation configuration."""
        default = {
            'min_engines_required': 2,
            'flag_threshold': 0.3
        }
        llm_config = self.config_data.get('llm', {})
        return llm_config.get('reconciliation', default)

    def get_metadata_config(self) -> Dict[str, bool]:
        """Get metadata extraction configuration."""
        default = {
            'extract_dates': True,
            'extract_authors': True,
            'extract_recipients': True,
            'preserve_structure': True
        }
        return self.config_data.get('metadata', default)

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        default = {
            'format': 'json',
            'output_dir': 'data/output',
            'keep_intermediate': True
        }
        return self.config_data.get('storage', default)

    def get_api_keys(self) -> Dict[str, str]:
        """Get API keys from environment."""
        return {
            'google_vision': os.environ.get('GOOGLE_CLOUD_VISION_CREDENTIALS_PATH'),
            'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
            'aws_region': os.environ.get('AWS_REGION', 'us-east-1'),
            'azure_key': os.environ.get('AZURE_COMPUTER_VISION_KEY'),
            'azure_endpoint': os.environ.get('AZURE_COMPUTER_VISION_ENDPOINT'),
            'openai_key': os.environ.get('OPENAI_API_KEY'),
            'anthropic_key': os.environ.get('ANTHROPIC_API_KEY'),
            'gemini_key': os.environ.get('GOOGLE_GEMINI_API_KEY')
        }
