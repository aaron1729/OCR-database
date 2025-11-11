# Claude Development Guide

This document provides context and instructions for AI assistants (particularly Claude) working on this OCR pipeline project.

## Project Overview

This is a sophisticated OCR pipeline designed to process historical handwritten documents (primarily letters from the 1800s) using multiple OCR engines and LLM-based reconciliation for maximum accuracy.

**Key Innovation**: Rather than relying on a single OCR engine, we run multiple engines in parallel (Tesseract, Google Vision, AWS Textract, Azure Vision) and use an LLM to intelligently reconcile the results, producing higher-quality transcriptions than any single engine could achieve.

## Architecture

### Core Philosophy

1. **Modular Design**: Each component (preprocessing, OCR, reconciliation, metadata, storage) is independent and extensible
2. **Provider Agnostic**: LLM reconciliation works with OpenAI, Anthropic, or Google Gemini
3. **Graceful Degradation**: Pipeline works with as few as 1 OCR engine if others are unavailable
4. **Preservation**: Option to keep all intermediate results for analysis and debugging

### Data Flow

```
PDF → Images → Enhanced Images → Multiple OCR Results → LLM Reconciliation → Structured JSON
       ↓                                                          ↓
   Metadata ←─────────────── Extracted Text ──────────────── Metadata
```

## Project Structure

```
src/
├── preprocessing/         # PDF → Images, Image Enhancement
│   ├── pdf_processor.py      # PDF handling, metadata extraction
│   └── image_enhancer.py     # Deskew, contrast, denoise, binarize
├── ocr_engines/          # OCR Engine Wrappers
│   ├── base.py               # Base classes: OCREngine, OCRResult, TextBlock
│   ├── tesseract_engine.py
│   ├── google_vision_engine.py
│   ├── aws_textract_engine.py
│   └── azure_vision_engine.py
├── reconciliation/       # LLM-based merging
│   ├── llm_providers.py      # OpenAIProvider, AnthropicProvider, GeminiProvider
│   └── llm_reconciler.py     # Discrepancy detection, reconciliation logic
├── metadata/             # Content analysis
│   └── extractor.py          # Extract dates, authors, recipients, locations
├── storage/              # Persistence
│   └── json_storage.py       # JSON-based storage and retrieval
├── config.py             # Configuration management (YAML + env vars)
└── pipeline.py           # Main orchestration
```

## Key Design Patterns

### 1. OCR Engine Interface

All OCR engines inherit from `OCREngine` base class and implement:
- `process_image(image, page_number) -> OCRResult`
- `is_available() -> bool`

This allows easy addition of new engines without changing the pipeline.

### 2. LLM Provider Interface

All LLM providers inherit from `LLMProvider` and implement:
- `generate(prompt, system_prompt) -> str`
- `is_available() -> bool`

This keeps reconciliation logic independent of the specific LLM being used.

### 3. Configuration Hierarchy

Configuration comes from two sources (in order of precedence):
1. Environment variables (`.env` file)
2. YAML config file (`config/example_config.yaml`)

API keys always come from environment variables for security.

## Common Development Tasks

### Adding a New OCR Engine

1. Create new file in `src/ocr_engines/` (e.g., `new_engine.py`)
2. Inherit from `OCREngine` base class
3. Implement `process_image()` and `is_available()`
4. Convert engine's output format to `OCRResult` with `TextBlock` objects
5. Add to `src/ocr_engines/__init__.py`
6. Update `pipeline.py` `_init_ocr_engines()` method
7. Add engine name to config options
8. Update documentation

Example skeleton:
```python
from .base import OCREngine, OCRResult, TextBlock, BoundingBox

class NewEngine(OCREngine):
    def __init__(self, api_key=None):
        super().__init__('new_engine')
        self.api_key = api_key

    def is_available(self):
        return self.api_key is not None

    def process_image(self, image, page_number=1):
        # Call your OCR API/library
        # Convert results to OCRResult format
        return OCRResult(...)
```

### Adding a New LLM Provider

1. Create provider class in `src/reconciliation/llm_providers.py`
2. Inherit from `LLMProvider`
3. Implement `generate()` and `is_available()`
4. Add to `__init__.py` exports
5. Update `pipeline.py` `_init_reconciliation()` to handle new provider
6. Update config options and documentation

### Improving Image Preprocessing

Edit `src/preprocessing/image_enhancer.py`:
- Add new enhancement method (follow existing pattern)
- Add configuration option to enable/disable it
- Update `enhance()` method to call new enhancement
- Document parameters in config YAML

### Enhancing Metadata Extraction

Edit `src/metadata/extractor.py`:
- Add new regex patterns or extraction logic
- Update `extract_from_text()` method
- Add new fields to `DocumentMetadata` dataclass
- Update storage serialization if needed

## Important Implementation Details

### 1. Image Format Conversion

The pipeline uses both PIL Images and OpenCV numpy arrays:
- PDFs are converted to PIL Images
- Image enhancement uses OpenCV (numpy arrays)
- OCR engines receive PIL Images
- Use `ImageEnhancer.pil_to_cv2()` and `cv2_to_pil()` for conversion

### 2. Confidence Scores

- All OCR engines normalize confidence to 0.0-1.0 range
- Tesseract returns 0-100, so divide by 100
- Some engines don't provide word-level confidence, use block average
- ReconciliationResult also includes an overall confidence score

### 3. Bounding Boxes

Different OCR engines return different coordinate systems:
- Tesseract: Pixel coordinates (absolute)
- Google Vision: Polygon vertices (absolute pixels)
- AWS Textract: Normalized 0-1 coordinates (convert to pixels using image size)
- Azure: Polygon points (absolute pixels)

All are converted to simple BoundingBox(x, y, width, height) for consistency.

### 4. LLM Reconciliation Prompt

The reconciliation prompt (`llm_reconciler.py`) is crucial for quality:
- Includes all OCR results with confidence scores
- Lists identified discrepancies
- Requests JSON response with specific schema
- Includes context about 1800s handwriting

If improving accuracy, focus on:
1. Better discrepancy detection (line-by-line comparison)
2. More detailed prompts with examples
3. Iterative reconciliation for complex cases

### 5. Error Handling

The pipeline should never crash due to a single engine failure:
- Each OCR engine wrapped in try/except
- Reconciliation has fallback to best single result
- Pipeline continues even if some pages fail
- All warnings logged but don't stop execution

## Testing Considerations

When testing the pipeline:

1. **Start with Tesseract only** - it's free and runs locally
2. **Use small test documents** first (1-2 pages)
3. **Check intermediate results** with `keep_intermediate: true`
4. **Verify API keys** are loaded correctly from `.env`
5. **Monitor API costs** - process test docs before large batches

### Sample Test Flow

```python
# Minimal test
config = Config()
pipeline = OCRPipeline(config)
pipeline.process_document('test.pdf', 'test_output')

# Check results
results = pipeline.storage.load_document('test_output')
print(results['metadata'])
print(results['pages'][0]['reconciliation']['merged_text'])
```

## Future Enhancement Ideas

### Phase 2: Searchability & RAG (Planned)

The user wants to add:
1. **Vector Database Integration**
   - Embed all processed documents
   - Store in Pinecone/Weaviate/ChromaDB
   - Enable semantic search

2. **RAG Query Interface**
   - Natural language questions
   - LLM retrieves relevant passages
   - Answers with citations

3. **Web Interface**
   - Upload PDFs
   - View processing status
   - Search and browse results

### Other Potential Improvements

- **Batch Processing**: Process multiple PDFs in parallel
- **Human Review Interface**: Flag uncertain transcriptions for manual review
- **Training Data Export**: Generate training data for fine-tuning OCR models
- **Quality Metrics**: Track accuracy improvements from multi-engine approach
- **Cost Optimization**: Smart engine selection based on document characteristics
- **Language Support**: Extend beyond English to other languages
- **Handwriting Style Analysis**: Identify authors by handwriting characteristics

## Debugging Tips

### OCR Engine Issues

```python
# Test individual engine
from src.ocr_engines import TesseractEngine
from PIL import Image

engine = TesseractEngine()
print(engine.is_available())  # Should be True

image = Image.open('test.png')
result = engine.process_image(image)
print(result.full_text)
print(result.average_confidence)
```

### LLM Provider Issues

```python
# Test provider directly
from src.reconciliation.llm_providers import OpenAIProvider

provider = OpenAIProvider()
print(provider.is_available())

response = provider.generate("Say hello", "You are a helpful assistant")
print(response)
```

### Configuration Issues

```python
# Check what's loaded
from src.config import Config

config = Config('config/example_config.yaml')
print(config.get_ocr_engines())
print(config.get_llm_config())
print(config.get_api_keys())
```

## Code Style Guidelines

- **Type hints**: Use typing annotations for all function parameters and returns
- **Docstrings**: All classes and public methods should have docstrings
- **Error messages**: Include context in exception messages
- **Logging**: Use print() for now, but structure for easy migration to logging module
- **Configuration**: Never hardcode values, use config system
- **Imports**: Use relative imports within src/ package

## Performance Considerations

- **PDF Conversion**: DPI setting (default 300) affects quality vs. speed
- **Image Enhancement**: Most expensive operations are deskewing and denoising
- **OCR Engines**: Run in sequence currently; could parallelize with multiprocessing
- **LLM Calls**: Most expensive part; consider batching pages for long documents
- **Storage**: JSON works for hundreds of documents; consider database for larger scale

## Security Notes

- Never commit `.env` file (in .gitignore)
- API keys should only come from environment variables
- Validate file paths to prevent directory traversal
- Be cautious with OCR text used in prompts (potential injection)

## Getting Help

If you (future Claude) need to understand something:
1. Read the relevant module's code - it's well-documented
2. Check `example_usage.py` for usage patterns
3. Look at `process_document.py` for CLI integration
4. Review tests/ directory (if tests added)
5. Consult the original libraries' documentation

## Benchmarking Results (November 2024)

We conducted extensive benchmarks on real 1860s handwritten letters to evaluate OCR accuracy:

### Test Documents
- **109-4-16, letters sent 1864 Atlanta** (20 pages)
- **109-4-31, 4-63--8-64 Macon Armory letters** (20 pages)

### Results Summary

| Engine | Atlanta Confidence | Macon Confidence | Speed |
|--------|-------------------|------------------|-------|
| **Google Vision API** | **65.8%** | **73.6%** | ~5-6s/page |
| Tesseract (300 DPI, binarized) | 32.5% | 34.0% | ~2-4s/page |
| Tesseract (300 DPI, enhanced) | 30.2% | 28.7% | ~3-4s/page |
| Tesseract (150 DPI, enhanced) | 30.2% | 28.1% | ~1-2s/page |
| PaddleOCR | 0% (API errors) | 0% (API errors) | N/A |

**Key Findings:**
- Google Vision is **2-2.2x more accurate** than Tesseract on 1860s handwriting
- Google Vision achieved 65-74% confidence on challenging cursive writing
- Best Tesseract config: 300 DPI with binarization (~32-34% confidence)
- Handwriting quality varies: Macon letters were more legible than Atlanta letters
- **Conclusion**: Automated OCR for 1860s handwriting is feasible with cloud-based engines

### Benchmark Output Structure

Benchmark results are saved to:
```
benchmark_results/
├── benchmark_results.json          # Full JSON results with per-page data
├── text_output/                    # Atlanta letters text files
│   ├── Google_Vision_API.txt
│   ├── Tesseract_300_DPI_binarized.txt
│   └── ...
└── text_output_macon/              # Macon letters text files
    ├── Google_Vision_API.txt
    └── ...
```

Each text file contains page-by-page OCR output with confidence scores for easy spot-checking against original PDFs.

### Running Benchmarks

Use `benchmark_ocr_engines.py` to test OCR engines:
```python
# Configure in the script:
pdf_path = 'path/to/test.pdf'

# Run benchmark
python benchmark_ocr_engines.py

# Results saved to benchmark_results/
```

The benchmark:
1. Converts PDF to images at multiple DPIs
2. Applies image enhancement (deskew, CLAHE, denoise, binarization)
3. Runs all available OCR engines
4. Reports confidence and speed per page
5. Ranks engines by accuracy

## Current Implementation Status

### ✅ Completed Components

- **PDF Processing**: pypdfium2-based (pure Python, no system dependencies)
- **Image Enhancement**: Deskewing, CLAHE, bilateral filtering, binarization
- **OCR Engines**:
  - ✅ Tesseract (local, free)
  - ✅ Google Cloud Vision (cloud, paid, best accuracy)
  - ✅ AWS Textract (not tested yet)
  - ✅ Azure Vision (not tested yet)
  - ⚠️ PaddleOCR (installed but API errors)
- **LLM Reconciliation**: Anthropic Claude, OpenAI, Google Gemini support
- **Metadata Extraction**: Regex-based extraction for dates, names, locations
- **Storage**: JSON-based document storage
- **Benchmarking**: Comprehensive OCR comparison tool

### 🚧 In Progress / Not Yet Tested

- Full pipeline integration (preprocessing → multi-OCR → reconciliation)
- AWS Textract and Azure Vision testing
- LLM reconciliation testing on real documents
- Cost analysis for Google Vision API at scale

### 📋 Planned (Phase 2)

- Vector database integration (Pinecone/Weaviate/ChromaDB)
- RAG-based semantic search
- Web interface for upload/search
- Batch processing

## API Keys & Configuration

Currently configured (in `.env`):
- `ANTHROPIC_API_KEY` - For LLM reconciliation (Claude)
- `GOOGLE_CLOUD_VISION_CREDENTIALS_PATH=credentials/google-cloud-vision.json` - For OCR

Not yet configured:
- `OPENAI_API_KEY` - Alternative LLM provider
- `GOOGLE_GEMINI_API_KEY` - Alternative LLM provider
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - For Textract OCR
- `AZURE_COMPUTER_VISION_KEY` / `AZURE_COMPUTER_VISION_ENDPOINT` - For Azure OCR

### Google Cloud Vision Setup

The project uses Google Cloud Vision API for high-accuracy handwriting OCR. To set up:

1. Create Google Cloud project
2. Enable Cloud Vision API
3. Create service account with "Cloud Vision API User" role
4. Download JSON credentials
5. Save to `credentials/google-cloud-vision.json`
6. Add to `.env`: `GOOGLE_CLOUD_VISION_CREDENTIALS_PATH=credentials/google-cloud-vision.json`

The `credentials/` directory is gitignored to prevent accidental credential commits.

## Version History

- **v1.0** (2024-11): Initial implementation
  - Multi-engine OCR pipeline
  - LLM reconciliation
  - Metadata extraction
  - JSON storage
  - CLI and programmatic interfaces
  - Comprehensive benchmarking tool
  - Proven feasibility on 1860s handwritten documents

---

**Remember**: The goal is high-accuracy transcription of difficult historical handwriting. Based on benchmarks, **Google Cloud Vision (65-74% confidence) combined with LLM reconciliation** is the recommended approach. Tesseract alone (~30% confidence) is insufficient for production use on cursive handwriting, but useful as a secondary source for reconciliation.
