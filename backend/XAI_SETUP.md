# xAI (Grok) Integration Setup

This guide explains how to set up and use xAI's Grok model instead of OpenAI in your Audio RAG system.

## Prerequisites

1. **xAI API Access**: You need access to xAI's API. Visit [x.ai](https://x.ai) to sign up for API access.
2. **API Key**: Get your API key from the xAI dashboard.

## Installation

1. **Install xAI SDK**:
   ```bash
   pip install xai-sdk
   ```

2. **Set Environment Variable**:
   ```bash
   # Windows
   set GROK_API_KEY=your_xai_api_key_here
   
   # Linux/Mac
   export GROK_API_KEY=your_xai_api_key_here
   ```

## Configuration

The system is now configured to use Grok instead of OpenAI:

- **Model**: `grok-3-mini-fast` (fast and efficient)
- **Temperature**: 0 (deterministic responses)
- **API Host**: `api.x.ai`

## Testing the Integration

Run the test script to verify everything works:

```bash
python test_xai_integration.py
```

## Usage

The Audio RAG system will now:

1. **Transcribe audio** using Whisper
2. **Search knowledge base** using FAISS
3. **Generate responses** using Grok instead of GPT-3.5-turbo

## API Changes

### Environment Variables
- **Old**: `OPENAI_API_KEY`
- **New**: `GROK_API_KEY`

### Health Check Response
The `/health` endpoint now shows:
```json
{
  "xai_configured": true,
  "chunking_config": {
    "model_name": "grok-3-mini-fast"
  }
}
```

## Troubleshooting

### Common Issues

1. **"xAI SDK not installed"**
   - Solution: `pip install xai-sdk`

2. **"API key not provided"**
   - Solution: Set `GROK_API_KEY` environment variable

3. **"Access denied"**
   - Solution: Check your xAI API access and key validity

4. **"Model not found"**
   - Solution: Ensure you're using `grok-3-mini-fast` (available model)

### Testing

If you encounter issues, run the test script:
```bash
python test_xai_integration.py
```

This will help identify specific problems with the xAI integration.

## Performance Notes

- **Grok-3-mini-fast** is optimized for speed and efficiency
- Response times may vary compared to OpenAI
- The model is well-suited for RAG applications

## Migration from OpenAI

If you're migrating from OpenAI:

1. Install xAI SDK: `pip install xai-sdk`
2. Set `GROK_API_KEY` instead of `OPENAI_API_KEY`