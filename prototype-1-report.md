# Audio RAG Pipeline - Prototype 1 Configuration Report

## Project Overview
This document provides a comprehensive overview of all configurations, components, and technologies used in the Audio RAG (Retrieval-Augmented Generation) pipeline prototype-1.

## Architecture Overview
The system consists of:
- **Frontend**: React Native mobile app with audio recording capabilities
- **Backend**: FastAPI server with audio processing, transcription, and RAG pipeline
- **Vector Database**: FAISS for document storage and similarity search
- **Knowledge Base**: Local document collection for context retrieval

---

## Backend Configuration

### Core Framework
- **Framework**: FastAPI 0.104.1
- **Server**: Uvicorn 0.24.0 (standard)
- **Language**: Python 3.x
- **Architecture**: RESTful API with async support

### Audio Processing Pipeline

#### 1. Audio Transcription
- **Engine**: OpenAI Whisper (20231117)
- **Model**: `whisper.load_model("base")`
- **Configuration**:
  - `fp16=False` (disabled half-precision for compatibility)
  - `language="en"` (English)
  - `task="transcribe"` (transcription task)
- **Fallback Methods**: Multiple transcription attempts with different parameters

#### 2. Audio Processing Components
- **Primary Library**: `audio_processor.py` (custom implementation)
- **Audio Conversion**: Multiple fallback methods:
  - Direct FFmpeg conversion
  - Librosa (0.10.1) for audio processing
  - Pydub (0.25.1) for format conversion
  - Soundfile (0.12.1) for audio I/O

#### 3. FFmpeg Configuration
- **FFmpeg Path**: `C:\ffmpeg\bin\ffmpeg.exe`
- **FFprobe Path**: `C:\ffmpeg\bin\ffprobe.exe`
- **Environment Variables**:
  - `FFMPEG_BINARY`: Set to FFmpeg executable path
  - `FFPROBE_BINARY`: Set to FFprobe executable path
  - `PATH`: Includes FFmpeg bin directory

### Vector Database & Embeddings

#### 1. Vector Store
- **Database**: FAISS (Facebook AI Similarity Search)
- **Version**: faiss-cpu 1.7.4
- **Index Type**: `IndexFlatIP` (Inner Product for cosine similarity)
- **Storage**: Local file-based storage
- **Files**: 
  - `faiss_knowledge_base.index` (FAISS index)
  - `faiss_knowledge_base.metadata` (document metadata)

#### 2. Embedding Model
- **Model**: Sentence Transformers
- **Version**: 2.2.2
- **Specific Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Normalization**: L2 normalization for cosine similarity
- **Batch Size**: 32 for embedding generation

#### 3. Vector Store Features
- **Search Results**: Top 3 most similar documents
- **Document Chunking**: Automatic splitting for documents >50KB
- **Metadata Storage**: Source file information and chunk identifiers
- **Persistence**: Automatic saving after each batch

### Language Model Integration

#### 1. OpenAI Integration
- **Library**: openai 1.3.7
- **Model**: GPT-3.5-turbo
- **Configuration**:
  - API Key: Environment variable `OPENAI_API_KEY`
  - Client: OpenAI client with API key authentication

#### 2. Prompt Engineering
- **Format**: Context-based question answering
- **Template**: `"Answer the question based on context: {contexts}\nQuestion: {query_text}"`
- **Context**: Retrieved documents from FAISS vector store

### Document Processing

#### 1. Knowledge Base
- **Location**: `backend/data/sample_kb/`
- **Format**: Markdown files (.md)
- **Content**: Transcript files from various sources

#### 2. Token-Based Document Chunking
- **Library**: Custom implementation using tiktoken (0.5.2)
- **Tokenizer**: tiktoken with GPT-3.5-turbo encoding
- **Chunk Size**: 512 tokens (configurable via `CHUNK_SIZE` environment variable)
- **Chunk Overlap**: 50 tokens (configurable via `CHUNK_OVERLAP` environment variable)
- **Splitter**: Custom token-based splitter with intelligent boundary detection
- **Features**:
  - Accurate token counting using tiktoken
  - Overlap between chunks for context preservation
  - Metadata tracking (chunk index, total chunks, token count)
  - Intelligent boundary detection at sentence boundaries
  - Word boundary preservation to avoid cutting words

#### 3. Document Preprocessing
- **Whitespace Normalization**: Remove excessive blank lines and spaces
- **Line Length Filtering**: Remove lines >2000 characters
- **Duplicate Removal**: Eliminate duplicate paragraphs
- **Token-Aware Processing**: Respects semantic boundaries

#### 4. Chunking Strategy
- **Primary**: Token-based splitting with overlap
- **Fallback**: Character-based splitting for edge cases
- **Metadata**: Rich chunk metadata for debugging and optimization
- **Configuration**: Environment variable driven for flexibility

### API Endpoints

#### 1. Query Endpoint
- **Path**: `/query`
- **Method**: POST
- **Input**: Audio file (multipart/form-data) or raw audio bytes
- **Output**: JSON with transcript, response, and latency metrics
- **Processing Steps**:
  1. Audio processing and transcription
  2. Vector store search
  3. LLM response generation
  4. Latency tracking

#### 2. Health Check Endpoint
- **Path**: `/health`
- **Method**: GET
- **Output**: System status and component availability

### Performance Optimizations

#### 1. Batch Processing
- **Document Ingestion**: 25 documents per batch
- **Embedding Generation**: Batch size of 32
- **Vector Store Saving**: After each batch

#### 2. Caching
- **Whisper Model**: Loaded once at startup
- **Vector Store**: Loaded once at startup
- **OpenAI Client**: Initialized once at startup

#### 3. Error Handling
- **Multiple Transcription Attempts**: 4 different methods
- **Graceful Degradation**: Fallback to sample context
- **Batch Continuation**: Continue processing on individual batch failures

---

## Frontend Configuration

### Core Framework
- **Framework**: React Native
- **Version**: 0.79.3
- **Language**: TypeScript 5.3.0
- **Platform**: Expo SDK 53.0.0

### Audio Recording

#### 1. Audio Library
- **Library**: Expo AV (15.1.6)
- **Capabilities**: Audio recording and playback
- **Permissions**: Microphone access required

#### 2. Recording Configuration
- **Format**: WAV
- **Sample Rate**: 16000 Hz
- **Channels**: 1 (mono)
- **Bit Rate**: 128000 bps
- **Quality**: HIGH_QUALITY preset

#### 3. Audio Processing
- **Encoding**: Base64 for transmission
- **Binary Conversion**: Uint8Array for backend communication
- **Content Type**: `application/octet-stream`

### UI Components

#### 1. Core Components
- **Recording Button**: Animated with pulse effect
- **Status Indicators**: Connection status, loading states
- **Text Display**: Transcript and response areas
- **Animations**: React Native Animated API

#### 2. Styling
- **Framework**: React Native StyleSheet
- **Icons**: Expo Vector Icons (Ionicons)
- **Responsive Design**: Platform-agnostic layout

### Network Configuration

#### 1. Backend Communication
- **Base URL**: `http://192.168.1.192:8001` (configurable)
- **Health Check**: Every 10 seconds
- **Error Handling**: Alert dialogs for user feedback

#### 2. API Integration
- **Query Endpoint**: POST to `/query`
- **Health Endpoint**: GET to `/health`
- **Headers**: Content-Type for audio data

---

## Dependencies

### Backend Dependencies (requirements.txt)
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
openai-whisper==20231117
faiss-cpu==1.7.4
openai==1.3.7
python-multipart==0.0.6
librosa==0.10.1
soundfile==0.12.1
pydub==0.25.1
numpy==1.24.3
sentence-transformers==2.2.2
scikit-learn==1.3.0
tiktoken==0.5.2
```

### Frontend Dependencies (package.json)
```json
{
  "@expo/vector-icons": "^14.1.0",
  "expo": "~53.0.0",
  "expo-av": "~15.1.6",
  "expo-file-system": "~18.1.10",
  "react": "^19.0.0",
  "react-native": "^0.79.3"
}
```

---

## System Requirements

### Backend Requirements
- **Python**: 3.8+
- **FFmpeg**: Installed at `C:\ffmpeg\bin\`
- **Memory**: Sufficient for Whisper model (~1GB)
- **Storage**: Space for vector store and knowledge base
- **Network**: Internet access for OpenAI API

### Frontend Requirements
- **Node.js**: For development
- **Expo CLI**: For app development
- **Mobile Device**: iOS/Android with microphone
- **Network**: WiFi connection to backend

---

## Configuration Files

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for LLM access
- `SKIP_INGESTION`: Set to "true" to skip document ingestion
- `CHROMA_TELEMETRY`: Disabled (not used with FAISS)
- `PATH`: Includes FFmpeg binary path
- `CHUNK_SIZE`: Token count per chunk (default: 512)
- `CHUNK_OVERLAP`: Overlapping tokens between chunks (default: 50)

### File Structure
```
backend/
├── main.py                 # FastAPI server
├── audio_processor.py      # Audio processing utilities
├── faiss_vector_store.py   # FAISS vector store implementation
├── requirements.txt        # Python dependencies
├── data/sample_kb/         # Knowledge base documents
└── faiss_knowledge_base.*  # Vector store files

frontend/
├── src/App.tsx            # Main React Native app
├── package.json           # Node.js dependencies
└── app.json              # Expo configuration
```

---

## Performance Metrics

### Latency Tracking
- **Total Latency**: End-to-end processing time
- **Retrieval Latency**: Vector store search time
- **LLM Latency**: OpenAI API response time
- **Audio Processing**: Transcription and conversion time

### Scalability Considerations
- **Vector Store**: FAISS supports millions of vectors
- **Document Processing**: Batch processing for large datasets
- **Memory Management**: Automatic cleanup of temporary files
- **Error Recovery**: Graceful handling of component failures

---

## Security Considerations

### API Security
- **CORS**: Configured to allow all origins (development)
- **Input Validation**: Audio file validation and size limits
- **Error Handling**: No sensitive information in error messages

### Data Privacy
- **Local Processing**: Audio processed locally before API calls
- **Temporary Files**: Automatic cleanup after processing
- **No Persistent Storage**: Audio files not stored permanently

---

## Known Limitations

### Current Limitations
1. **FFmpeg Dependency**: Requires manual FFmpeg installation
2. **Single Language**: Optimized for English transcription
3. **Model Size**: Using "base" Whisper model (accuracy vs. speed trade-off)
4. **Network Dependency**: Requires internet for OpenAI API
5. **Local Knowledge Base**: Limited to pre-loaded documents

### Future Improvements
1. **Multi-language Support**: Expand to other languages
2. **Model Optimization**: Use larger Whisper models for better accuracy
3. **Real-time Processing**: Stream audio for lower latency
4. **Distributed Architecture**: Scale across multiple servers
5. **Advanced RAG**: Implement more sophisticated retrieval strategies

---

## Deployment Notes

### Development Setup
1. Install Python dependencies: `pip install -r requirements.txt`
2. Install FFmpeg at `C:\ffmpeg\bin\`
3. Set OpenAI API key environment variable
4. Run backend: `uvicorn main:app --host 0.0.0.0 --port 8001`
5. Run frontend: `expo start`

### Production Considerations
1. **Environment Variables**: Secure API key management
2. **CORS Configuration**: Restrict to specific origins
3. **Logging**: Implement proper logging and monitoring
4. **Error Handling**: Comprehensive error tracking
5. **Performance Monitoring**: Track latency and throughput metrics

---

*Report generated for Audio RAG Pipeline - Prototype 1*
*Date: January 2025*

Knowledge Sources
├── transcript_BayICT Tech Talk - Bharani Rajakumar (Chunk 1)
│   From: transcript_BayICT Tech Talk - Bharani Rajakumar.md
│   Relevance: 85.0%
├── transcript_Episode 2 OKRs vs KPIs (Chunk 2)
│   From: transcript_Episode 2 OKRs vs KPIs.md
│   Relevance: 72.0%
└── transcript_WorkingNation Overheard Bharani Rajakumar on VR (Chunk 1)
    From: transcript_WorkingNation Overheard Bharani Rajakumar on VR as a pathway to middle-skill jobs.md
    Relevance: 68.0% 