# Audio RAG Backend Setup Guide

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- Virtual environment (recommended)

### 2. FFmpeg Installation (Required for Audio Processing)

#### Windows:
1. **Download FFmpeg:**
   - Go to https://ffmpeg.org/download.html
   - Download the Windows builds from https://github.com/BtbN/FFmpeg-Builds/releases
   - Choose the latest release with "essentials" build

2. **Install FFmpeg:**
   - Extract the downloaded zip file
   - Copy the extracted folder to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to your system PATH:
     - Open System Properties → Advanced → Environment Variables
     - Edit the PATH variable
     - Add `C:\ffmpeg\bin`
     - Restart your terminal/command prompt

3. **Verify Installation:**
   ```bash
   ffmpeg -version
   ```

#### Alternative: Using Chocolatey (Windows):
```bash
choco install ffmpeg
```

#### Alternative: Using Scoop (Windows):
```bash
scoop install ffmpeg
```

### 3. Environment Variables
Create a `.env` file in the backend directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
RESET_DB=false
```

## Installation Steps

### 1. Create Virtual Environment
```bash
cd backend
python -m venv venv
```

### 2. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import whisper; print('Whisper installed successfully')"
python -c "import librosa; print('Librosa installed successfully')"
python -c "import soundfile; print('Soundfile installed successfully')"
python -c "from pydub import AudioSegment; print('Pydub installed successfully')"
```

## Running the Backend

### 1. Start the Server
```bash
# From the backend directory
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### 2. Test the Health Endpoint
```bash
curl http://localhost:8001/health
```

## Troubleshooting

### Common Issues

#### 1. FFmpeg Not Found
**Error:** `Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work`

**Solution:**
- Ensure FFmpeg is installed and in your PATH
- Restart your terminal after adding FFmpeg to PATH
- Verify with: `ffmpeg -version`

#### 2. Audio Processing Failures
**Error:** `The system cannot find the file specified`

**Solutions:**
- The improved code now has multiple fallback methods
- Check that audio files are not corrupted
- Ensure audio format is supported (WAV, MP3, M4A, etc.)
- Verify audio contains speech content

#### 3. Whisper Model Loading Issues
**Error:** `Error loading Whisper model`

**Solutions:**
- Ensure sufficient disk space (Whisper models are ~1GB)
- Check internet connection for model download
- Try using a smaller model: `whisper.load_model("tiny")`

#### 4. ChromaDB Issues
**Error:** `Error initializing ChromaDB`

**Solutions:**
- Ensure write permissions in the backend directory
- Check available disk space
- Try resetting the database: `RESET_DB=true`

### Audio Format Support

The system supports multiple audio formats:
- **WAV** (recommended)
- **MP3**
- **M4A**
- **FLAC**
- **OGG**

### Performance Tips

1. **Use WAV format** for best compatibility
2. **Keep audio files under 10MB** for faster processing
3. **Ensure good audio quality** (16kHz+ sample rate, clear speech)
4. **Use mono audio** when possible

## Testing

### Test Audio Processing
```bash
# Test with a sample audio file
curl -X POST http://localhost:8001/query \
  -F "audio=@path/to/your/audio.wav"
```

### Test Health Check
```bash
curl http://localhost:8001/health
```

## Development

### Adding New Audio Processing Methods
The code includes multiple fallback methods for audio processing:
1. Direct Whisper transcription
2. Parameter-adjusted Whisper transcription
3. Librosa audio conversion
4. Pydub audio conversion
5. Numpy array transcription

### Debugging
Enable detailed logging by checking the console output for:
- File creation and deletion
- Audio conversion steps
- Transcription attempts
- Error messages

## Support

If you encounter issues:
1. Check the console output for detailed error messages
2. Verify all prerequisites are installed
3. Test with a simple WAV file first
4. Check the health endpoint for system status 