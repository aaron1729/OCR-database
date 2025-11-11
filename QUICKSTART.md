# Quick Start Guide

Get up and running with the OCR pipeline in 5 minutes!

## Minimal Setup (Tesseract + OpenAI)

This guide shows the fastest way to get started with just Tesseract (free) and OpenAI for reconciliation.

### 1. Install System Dependencies

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

### 2. Set Up Python Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install minimal dependencies
pip install pytesseract pillow pdf2image opencv-python numpy scikit-image
pip install openai python-dotenv pyyaml tqdm
```

### 3. Configure API Keys

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your OpenAI key
# OPENAI_API_KEY=sk-...
```

### 4. Create Minimal Config

Create `config/quickstart_config.yaml`:
```yaml
ocr:
  engines:
    - tesseract

llm:
  provider: openai
  model: gpt-4o-mini  # Cheaper model for testing
  temperature: 0.2

storage:
  output_dir: data/output
  keep_intermediate: false
```

### 5. Process Your First Document

```bash
python process_document.py data/input/your_document.pdf -c config/quickstart_config.yaml
```

### 6. View Results

```python
from src.storage import JSONStorage

storage = JSONStorage('data/output')
text = storage.get_full_text('your_document')
print(text)
```

## Adding More OCR Engines

### Google Cloud Vision

1. Create a Google Cloud project and enable Vision API
2. Download credentials JSON
3. Update `.env`:
   ```
   GOOGLE_CLOUD_VISION_CREDENTIALS_PATH=/path/to/credentials.json
   ```
4. Install: `pip install google-cloud-vision`
5. Add to config:
   ```yaml
   ocr:
     engines:
       - tesseract
       - google_vision
   ```

### AWS Textract

1. Create AWS account and get access keys
2. Update `.env`:
   ```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   AWS_REGION=us-east-1
   ```
3. Install: `pip install boto3`
4. Add to config:
   ```yaml
   ocr:
     engines:
       - tesseract
       - aws_textract
   ```

### Azure Computer Vision

1. Create Azure Cognitive Services resource
2. Update `.env`:
   ```
   AZURE_COMPUTER_VISION_KEY=your_key
   AZURE_COMPUTER_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   ```
3. Install: `pip install azure-cognitiveservices-vision-computervision`
4. Add to config:
   ```yaml
   ocr:
     engines:
       - tesseract
       - azure_vision
   ```

## Using Different LLM Providers

### Anthropic Claude

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...

# config
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
```

Install: `pip install anthropic`

### Google Gemini

```bash
# .env
GOOGLE_GEMINI_API_KEY=...

# config
llm:
  provider: gemini
  model: gemini-pro
```

Install: `pip install google-generativeai`

## Troubleshooting

### "Tesseract not found"
- Make sure Tesseract is installed: `tesseract --version`
- On Windows, add Tesseract to PATH

### "No module named 'cv2'"
- Install OpenCV: `pip install opencv-python`

### "PDF conversion failed"
- Install Poppler (see system dependencies above)

### "LLM provider not available"
- Check your API key in `.env`
- Verify the key is valid and has credits

### Import errors
- Install all requirements: `pip install -r requirements.txt`
- Make sure virtual environment is activated

## Next Steps

- Review full documentation in README.md
- Customize configuration in `config/example_config.yaml`
- Try `example_usage.py` for programmatic usage
- Process your historical documents!

## Cost Considerations

- **Tesseract**: Free (runs locally)
- **Google Vision**: ~$1.50 per 1000 pages
- **AWS Textract**: ~$1.50 per 1000 pages
- **Azure Vision**: ~$1.00 per 1000 pages
- **OpenAI GPT-4o**: ~$2.50 per 1M input tokens
- **Anthropic Claude**: ~$3.00 per 1M input tokens
- **Google Gemini**: Free tier available

For testing, start with just Tesseract + cheapest LLM model!
