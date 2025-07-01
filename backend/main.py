import os
# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")
    pass

# 1) tell ChromaDB at import time to disable *all* telemetry
# os.environ["CHROMA_TELEMETRY"] = "false"  # No longer needed with Qdrant

# 2) optionally bump the PostHog logger so you never see it even if it starts up
import logging
# logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)  # No longer needed
logging.getLogger("qdrant_client").setLevel(logging.INFO)

# Ensure ffmpeg/ffprobe are in PATH for all subprocesses (including Whisper)
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ["PATH"]

import asyncio
import time
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from xai_sdk import Client
from xai_sdk.chat import user, system
from audio_processor import process_audio_for_whisper, AudioProcessor
from qdrant_vector_store import QdrantVectorStore
from text_chunker import create_chunker

import uuid
import typing
import threading
import signal
import re

# Store for session data
session_store = {}
# Store for chunk queues
chunk_queues = {}

# Optional imports - prioritize faster-whisper over standard whisper
try:
    from faster_whisper import WhisperModel
    fast_whisper_available = True
    print("✅ Faster Whisper available - will use for faster transcription")
except ImportError:
    WhisperModel = None
    fast_whisper_available = False
    print("⚠️ Faster Whisper not available, falling back to standard Whisper")

try:
    import whisper
    whisper_available = True
except ImportError:
    whisper = None
    whisper_available = False

try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system
except ImportError:
    Client = None
    user = None
    system = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KB_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_kb")

# Global variables for caching
vector_store = None
whisper_model = None
xai_client = None
startup_complete = False
chunker = None
using_fast_whisper = False

# Configuration
SKIP_INGESTION = os.getenv("SKIP_INGESTION", "false").lower() == "true"
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "qdrant_storage")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

def preprocess_document(content):
    """
    Preprocess document content to improve embedding performance and quality.
    This is now handled by the TokenBasedChunker class.
    """
    # Basic preprocessing - detailed preprocessing is done in the chunker
    content = content.strip()
    return content

def split_large_document(content, max_chunk_size=50000):
    """
    Split large documents into smaller chunks to prevent memory issues.
    This function is deprecated in favor of token-based chunking.
    Returns a list of document chunks.
    """
    if len(content) <= max_chunk_size:
        return [content]
    
    chunks = []
    # Split by paragraphs first
    paragraphs = content.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Single paragraph is too large, split by sentences
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            # Single sentence is too large, split by words
                            words = sentence.split()
                            for word in words:
                                if len(current_chunk) + len(word) + 1 > max_chunk_size:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                        current_chunk = word
                                    else:
                                        chunks.append(word)
                                else:
                                    current_chunk += " " + word if current_chunk else word
                    else:
                        current_chunk += ". " + sentence if current_chunk else sentence
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Load xAI API key and initialize client
api_key = os.getenv("GROK_API_KEY")
if Client is not None and api_key:
    try:
        xai_client = Client(
            api_host="api.x.ai",
            api_key=api_key
        )
        print("✅ xAI client initialized successfully")
    except Exception as e:
        print(f"Error initializing xAI client: {e}")
        xai_client = None
else:
    print("Warning: xAI SDK not installed or API key not provided")

@app.on_event("startup")
async def startup_event():
    global vector_store, whisper_model, startup_complete, chunker, using_fast_whisper
    print("Starting up Audio RAG system with Qdrant...")
    
    # Initialize token-based chunker
    print(f"Initializing token-based chunker (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    try:
        chunker = create_chunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print("✅ Token-based chunker initialized successfully")
    except Exception as e:
        print(f"Error initializing chunker: {e}")
        chunker = None
    
    # Initialize Whisper model - prioritize Fast Whisper for better performance
    if fast_whisper_available and WhisperModel is not None:
        print("Loading Fast Whisper model for optimized performance...")
        try:
            whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            using_fast_whisper = True
            print("✅ Fast Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Fast Whisper model: {e}")
            whisper_model = None
            using_fast_whisper = False
    elif whisper_available and whisper is not None:
        print("Loading standard Whisper model...")
        try:
            whisper_model = whisper.load_model("base")
            using_fast_whisper = False
            print("✅ Standard Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading standard Whisper model: {e}")
            whisper_model = None
            using_fast_whisper = False
    else:
        print("⚠️ Warning: No Whisper implementation available")
        whisper_model = None
        using_fast_whisper = False
    
    print("Initializing Qdrant vector store...")
    try:
        # Initialize Qdrant vector store with persistent storage
        print("Creating Qdrant vector store with persistent storage...")
        vector_store = QdrantVectorStore(
            storage_path=VECTOR_STORE_PATH,
            use_persistent_storage=True
        )
        print(f"✅ Qdrant vector store initialized with {vector_store.count()} documents")
            
    except Exception as e:
        print(f"Error initializing Qdrant vector store: {e}")
        import traceback
        traceback.print_exc()
        vector_store = None
        startup_complete = True
        return

    print("Forcing ingestion of KB...")
    if SKIP_INGESTION:
        print("SKIP_INGESTION is set to true, skipping document ingestion...")
    else:
        try:
            ingest_kb()
        except Exception as e:
            print(f"Error during document ingestion: {e}")

    startup_complete = True
    print("🏁 Startup complete!")

def ingest_kb():
    if vector_store is None:
        print("Qdrant vector store not available, skipping ingestion")
        return
    
    if chunker is None:
        print("Token-based chunker not available, skipping ingestion")
        return
        
    print("Starting document ingestion with token-based chunking...")
    print("Reading documents from knowledge base...")
    
    if not os.path.exists(KB_DIR):
        print(f"Knowledge base directory {KB_DIR} does not exist")
        return
    print(f"Knowledge base directory exists: {KB_DIR}")
    files = os.listdir(KB_DIR)
    print(f"Found {len(files)} files: {files}")
    
    # Process documents in smaller batches for better reliability
    BATCH_SIZE = 25  # Reduced from 50 to 25 for better reliability
    total_processed = 0
    total_chunks = 0
    
    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i + BATCH_SIZE]
        print(f"\n--- Processing batch {i//BATCH_SIZE + 1} ({len(batch_files)} files) ---")
        
        docs, metadatas, ids = [], [], []
        
        for fname in batch_files:
            path = os.path.join(KB_DIR, fname)
            print(f"Reading file: {fname}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Get chunking info for logging
                chunk_info = chunker.get_chunk_info(content)
                print(f"‣ Document tokens: {chunk_info['total_tokens']}, estimated chunks: {chunk_info['estimated_chunks']}")
                
                # Use token-based chunking
                chunk_objects = chunker.chunk_document(content, {"source": fname, "original_file": fname})
                
                if chunk_objects:
                    print(f"‣ Created {len(chunk_objects)} chunks")
                    for chunk_obj in chunk_objects:
                        docs.append(chunk_obj["content"])
                        metadatas.append(chunk_obj["metadata"])
                        ids.append(str(uuid.uuid4()))
                        total_chunks += 1
                else:
                    print(f"‣ No chunks created for {fname}")
                    
            except Exception as e:
                print(f"Error reading {fname}: {e}")
        
        if docs:
            print(f"Adding {len(docs)} chunks to vector store (batch {i//BATCH_SIZE + 1})...")
            
            try:
                print(f"Starting embedding for batch {i//BATCH_SIZE + 1}...")
                start_time = time.time()
                
                # Add documents to Qdrant vector store
                vector_store.add_documents(docs, metadatas, ids)
                
                end_time = time.time()
                print(f"✓ Added {len(docs)} chunks in {end_time - start_time:.2f}s")
                total_processed += len(docs)
                
                # Save vector store after each batch
                try:
                    vector_store.save(VECTOR_STORE_PATH)
                    print(f"✓ Saved vector store after batch {i//BATCH_SIZE + 1}")
                except Exception as e:
                    print(f"Warning: Could not save vector store: {e}")
                
            except Exception as e:
                print(f"✗ Error during vector store addition for batch {i//BATCH_SIZE + 1}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next batch instead of failing completely
                continue
        else:
            print(f"No documents in batch {i//BATCH_SIZE + 1}")
    
    print(f"\n--- Ingestion Complete ---")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total documents processed: {total_processed}")
    if total_processed > 0:
        print("Successfully ingested documents with token-based chunking")
        # Final save
        try:
            vector_store.save(VECTOR_STORE_PATH)
            print("✓ Final vector store save completed")
        except Exception as e:
            print(f"Warning: Could not save vector store: {e}")
    else:
        print("No documents were ingested")

def transcribe_audio(audio_path, whisper_model, using_fast_whisper=False):
    """
    Transcribe audio using either Fast Whisper or standard Whisper.
    Returns the transcribed text or None if transcription fails.
    """
    try:
        if using_fast_whisper:
            # Fast Whisper API
            segments, info = whisper_model.transcribe(audio_path, beam_size=5)
            # Fast Whisper returns segments, we need to concatenate them
            query_text = " ".join([segment.text for segment in segments])
            return query_text.strip() if query_text else None
        else:
            # Standard Whisper API
            result = whisper_model.transcribe(audio_path, fp16=False)
            query_text = result.get("text", "")
            return query_text.strip() if query_text else None
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

@app.post("/query")
async def query(request: Request, audio: UploadFile = File(None)):
    start_time = time.time()
    audio_processing_start = time.time()
    
    try:
        if whisper_model is None:
            return JSONResponse({"error": "whisper not available"}, status_code=500)
        
        # Handle both file upload and raw audio data
        audio_data = None
        if audio is not None:
            # File upload case
            audio_data = await audio.read()
        else:
            # Raw audio data case (from frontend)
            try:
                audio_data = await request.body()
            except Exception as e:
                print(f"Error reading request body: {e}")
                return JSONResponse({"error": f"Failed to read audio data: {e}"}, status_code=400)
        
        if not audio_data:
            return JSONResponse({"error": "No audio data provided"}, status_code=400)
        
        print(f"Received audio data of size: {len(audio_data)} bytes")
        
        # Process audio using our custom processor
        temp_path = None
        converted_path = None
        
        try:
            # Process audio for Whisper
            temp_path, converted_path = process_audio_for_whisper(audio_data)
            
            # Get audio info for debugging
            if temp_path:
                audio_info = AudioProcessor.get_audio_info(temp_path)
                print(f"Audio file info: {audio_info}")
            
            query_text = ""
            transcription_success = False
            
            # Method 1: Try with original file
            if not transcription_success and temp_path:
                try:
                    import os
                    print("File exists before transcription (original):", os.path.exists(temp_path), temp_path)
                    print("File size (original):", os.path.getsize(temp_path) if os.path.exists(temp_path) else 0)
                    print("Attempting transcription with original file...")
                    # Convert backslashes to forward slashes for Whisper
                    whisper_path = temp_path.replace("\\", "/")
                    query_text = transcribe_audio(whisper_path, whisper_model, using_fast_whisper)
                    if query_text:
                        print(f"Original file transcription successful: {query_text}")
                        transcription_success = True
                    else:
                        print("Original file transcription returned empty text")
                except Exception as e:
                    print(f"Original file transcription failed: {e}")
            
            # Method 2: Try with converted file (if available)
            if not transcription_success and converted_path:
                try:
                    import os
                    print("File exists before transcription (converted):", os.path.exists(converted_path), converted_path)
                    print("File size (converted):", os.path.getsize(converted_path) if os.path.exists(converted_path) else 0)
                    print("Attempting transcription with converted file...")
                    # Convert backslashes to forward slashes for Whisper
                    whisper_path = converted_path.replace("\\", "/")
                    query_text = transcribe_audio(whisper_path, whisper_model, using_fast_whisper)
                    if query_text:
                        print(f"Converted file transcription successful: {query_text}")
                        transcription_success = True
                    else:
                        print("Converted file transcription returned empty text")
                except Exception as e:
                    print(f"Converted file transcription failed: {e}")
            
            # Method 3: Try with different parameters (only for standard Whisper)
            if not transcription_success and temp_path and not using_fast_whisper:
                try:
                    print("Attempting transcription with adjusted parameters...")
                    result = whisper_model.transcribe(
                        temp_path, 
                        language="en",
                        task="transcribe"
                    )
                    query_text = result.get("text", "")
                    if query_text and isinstance(query_text, str):
                        query_text = query_text.strip()
                        if query_text:
                            print(f"Parameter-adjusted transcription successful: {query_text}")
                            transcription_success = True
                        else:
                            print("Parameter-adjusted transcription returned empty text")
                    else:
                        print("Parameter-adjusted transcription returned invalid text")
                except Exception as e:
                    print(f"Parameter-adjusted transcription failed: {e}")
            
            # Method 4: Try with numpy array (only for standard Whisper)
            if not transcription_success and temp_path and not using_fast_whisper:
                try:
                    print("Attempting transcription with numpy array...")
                    import librosa
                    
                    # Load audio as numpy array
                    audio_array, sample_rate = librosa.load(temp_path, sr=16000)
                    
                    # Try to transcribe directly from numpy array
                    result = whisper_model.transcribe(audio_array)
                    query_text = result.get("text", "")
                    if query_text and isinstance(query_text, str):
                        query_text = query_text.strip()
                        if query_text:
                            print(f"Numpy array transcription successful: {query_text}")
                            transcription_success = True
                        else:
                            print("Numpy array transcription returned empty text")
                    else:
                        print("Numpy array transcription returned invalid text")
                        
                except Exception as e:
                    print(f"Numpy array transcription failed: {e}")
            
            # If all methods failed
            if not transcription_success:
                error_msg = "All transcription methods failed. Please ensure:"
                error_msg += "\n1. Audio file is not corrupted"
                error_msg += "\n2. Audio format is supported (WAV, MP3, M4A, etc.)"
                error_msg += "\n3. Audio contains speech content"
                error_msg += "\n4. Audio quality is sufficient for transcription"
                error_msg += "\n5. Install FFmpeg for better audio format support"
                return JSONResponse({"error": error_msg}, status_code=500)
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": f"Audio processing failed: {e}"}, status_code=500)
        finally:
            # Clean up temp files
            cleanup_paths = [path for path in [temp_path, converted_path] if path is not None]
            if cleanup_paths:
                AudioProcessor.cleanup_temp_files(*cleanup_paths)
        
        audio_processing_latency = time.time() - audio_processing_start
        retrieval_start = time.time()
        contexts = []
        search_results = None
        
        # Use Qdrant vector store for retrieval
        if vector_store is not None:
            try:
                # Try to get relevant contexts from the knowledge base
                print(f"Querying Qdrant vector store with text: {query_text}")
                print(f"Vector store info: {vector_store.get_info()}")
                
                print("About to call vector_store.count()...")
                try:
                    count = vector_store.count()
                    print(f"Vector store count: {count}")
                except Exception as e:
                    print(f"Error getting vector store count: {e}")
                    contexts = ["Sample context for testing"]
                    return {
                        "transcript": query_text,
                        "response": "Error getting vector store count, using sample context",
                        "audio_processing_latency": audio_processing_latency,
                        "retrieval_latency": time.time() - retrieval_start,
                        "llm_latency": 0,
                        "total_latency": time.time() - start_time,
                        "retrieved_documents": [{"source": "Sample Context", "original_file": "Sample Context", "similarity_score": 0}],
                    }
                
                # Search for similar documents
                print("Searching Qdrant vector store...")
                try:
                    # Ensure query_text is a string
                    if isinstance(query_text, list):
                        query_text = " ".join(query_text)
                    elif not isinstance(query_text, str):
                        query_text = str(query_text)
                    
                    search_results = vector_store.search(query_text, n_results=3)
                    print(f"Qdrant search results: {search_results}")
                    if search_results and search_results['documents']:
                        contexts = search_results['documents']
                    else:
                        contexts = ["Sample context for testing"]
                        
                except Exception as e:
                    print(f"Qdrant search error: {e}")
                    contexts = ["Sample context for testing"]
                        
            except Exception as e:
                print(f"Error retrieving contexts: {e}")
                import traceback
                traceback.print_exc()
                contexts = ["Sample context for testing"]
        else:
            print("Qdrant vector store is None, using sample context")
            contexts = ["Sample context for testing"]
        
        retrieval_latency = time.time() - retrieval_start
        llm_start = time.time()
        response_text = ""
        retrieved_documents = []
        
        if xai_client is not None and system is not None and user is not None:
            try:
                # Create a new chat session with Grok
                chat = xai_client.chat.create(model="grok-3-mini-fast", temperature=0)
                
                # Add system message for context
                system_prompt = f"You are a helpful AI assistant. Answer the question based on the provided context. Only give the response and directly answer the questions Context, keep the answer short: {str(contexts)}"
                chat.append(system(system_prompt))
                
                # Add user question
                chat.append(user(str(query_text)))
                
                # Get the response using streaming for faster initial response
                response_text = ""
                try:
                    # Try streaming first for faster response
                    print("Starting streaming response...")
                    chunk_count = 0
                    for chunk in chat.stream():
                        chunk_count += 1
                        print(f"Received chunk {chunk_count}: {chunk}")
                        # Handle the streaming response properly
                        chunk_content = ""
                        if isinstance(chunk, tuple) and len(chunk) > 1:
                            # Handle tuple response (response, chunk)
                            chunk_data = chunk[1]
                            if hasattr(chunk_data, 'content') and chunk_data.content:
                                chunk_content = chunk_data.content
                                print(f"Added tuple chunk content: {chunk_data.content}")
                        elif hasattr(chunk, 'content') and chunk.content:
                            chunk_content = chunk.content
                            print(f"Added chunk content: {chunk.content}")
                        else:
                            print(f"Chunk has no content: {type(chunk)}")
                        
                        if chunk_content:
                            response_text += chunk_content
                    print(f"Streaming completed with {chunk_count} chunks")
                except Exception as stream_error:
                    print(f"Streaming failed, falling back to regular response: {stream_error}")
                    # Fallback to regular response
                    response = chat.sample()
                    response_text = response.content
                
            except Exception as e:
                print(f"xAI API error: {e}")
                response_text = f"Error: {e}"
        else:
            response_text = "LLM backend not configured"
        llm_latency = time.time() - llm_start
        
        # Prepare retrieved documents info for frontend
        if search_results and search_results['metadatas']:
            for i, metadata in enumerate(search_results['metadatas']):
                if i < 3:  # Limit to top 3 documents
                    doc_info = {
                        "source": metadata.get("source", "Unknown"),
                        "original_file": metadata.get("original_file", metadata.get("source", "Unknown")),
                        "similarity_score": search_results['distances'][i] if i < len(search_results['distances']) else 0
                    }
                    retrieved_documents.append(doc_info)
        else:
            # Fallback for sample context
            retrieved_documents = [{"source": "Sample Context", "original_file": "Sample Context", "similarity_score": 0}]
        
        total_latency = time.time() - start_time
        print(f"Query completed successfully. Total latency: {total_latency:.2f}s")
        print(f"Breakdown - Audio: {audio_processing_latency:.2f}s, Retrieval: {retrieval_latency:.2f}s, LLM: {llm_latency:.2f}s")
        print(f"Retrieved documents: {[doc['source'] for doc in retrieved_documents]}")
        
        return {
            "transcript": query_text,
            "response": response_text,
            "audio_processing_latency": audio_processing_latency,
            "retrieval_latency": retrieval_latency,
            "llm_latency": llm_latency,
            "total_latency": total_latency,
            "retrieved_documents": retrieved_documents,
        }
    except Exception as e:
        print(f"Unexpected error in query endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Unexpected error: {e}"}, status_code=500)

@app.post("/query/stream")
async def query_stream(request: Request, audio: UploadFile = File(None)):
    """
    Streaming endpoint that returns responses in real-time using Server-Sent Events (SSE).
    This provides faster perceived response times as the frontend receives chunks as they're generated.
    """
    async def generate_stream():
        start_time = time.time()
        audio_processing_start = time.time()
        
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing audio...'})}\n\n"
            
            if whisper_model is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Whisper not available'})}\n\n"
                return
            
            # Handle audio data
            audio_data = None
            if audio is not None:
                audio_data = await audio.read()
            else:
                try:
                    audio_data = await request.body()
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to read audio data: {e}'})}\n\n"
                    return
            
            if not audio_data:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No audio data provided'})}\n\n"
                return
            
            # Process audio
            temp_path = None
            converted_path = None
            
            try:
                temp_path, converted_path = process_audio_for_whisper(audio_data)
                query_text = transcribe_audio(temp_path, whisper_model, using_fast_whisper)
                
                if not query_text:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Transcription failed'})}\n\n"
                    return
                
                audio_processing_latency = time.time() - audio_processing_start
                yield f"data: {json.dumps({'type': 'transcript', 'text': query_text, 'latency': audio_processing_latency})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Audio processing failed: {e}'})}\n\n"
                return
            finally:
                if temp_path or converted_path:
                    cleanup_paths = [path for path in [temp_path, converted_path] if path is not None]
                    if cleanup_paths:
                        AudioProcessor.cleanup_temp_files(*cleanup_paths)
            
            # Retrieve context
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
            
            retrieval_start = time.time()
            contexts = []
            
            if vector_store is not None:
                try:
                    search_results = vector_store.search(query_text, n_results=3)
                    if search_results and search_results['documents']:
                        contexts = search_results['documents']
                    else:
                        contexts = ["Sample context for testing"]
                except Exception as e:
                    contexts = ["Sample context for testing"]
            else:
                contexts = ["Sample context for testing"]
            
            retrieval_latency = time.time() - retrieval_start
            yield f"data: {json.dumps({'type': 'retrieval_complete', 'latency': retrieval_latency})}\n\n"
            
            # Generate streaming response
            if xai_client is not None and system is not None and user is not None:
                try:
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
                    
                    chat = xai_client.chat.create(model="grok-3-mini-fast", temperature=0)
                    system_prompt = f"You are a helpful AI assistant. Answer the question based on the provided context. Only give the response and directly answer the questions Context, keep the answer short: {str(contexts)}"
                    chat.append(system(system_prompt))
                    chat.append(user(str(query_text)))
                    
                    llm_start = time.time()
                    response_text = ""
                    chunk_count = 0
                    
                    try:
                        for chunk in chat.stream():
                            chunk_count += 1
                            chunk_content = ""
                            
                            if isinstance(chunk, tuple) and len(chunk) > 1:
                                chunk_data = chunk[1]
                                if hasattr(chunk_data, 'content') and chunk_data.content:
                                    chunk_content = chunk_data.content
                            elif hasattr(chunk, 'content') and chunk.content:
                                chunk_content = chunk.content
                            
                            if chunk_content:
                                response_text += chunk_content
                                # Send chunk immediately
                                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_content, 'chunk_number': chunk_count})}\n\n"
                                print(f"🟢 Chunk {chunk_count} sent: {chunk_content}")
                                
                                # Small delay to make streaming visible
                                await asyncio.sleep(0.005)
                    except Exception as stream_error:
                        print(f"Streaming failed, falling back to regular response: {stream_error}")
                        # Fallback to regular response
                        response = chat.sample()
                        response_text = response.content
                    
                    llm_latency = time.time() - llm_start
                    total_latency = time.time() - start_time
                    
                    # Send final response with metadata
                    yield f"data: {json.dumps({'type': 'complete', 'full_response': response_text, 'llm_latency': llm_latency, 'total_latency': total_latency, 'total_chunks': chunk_count})}\n\n"
                        
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'LLM error: {e}'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'LLM backend not configured'})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Unexpected error: {e}'})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.post("/query/hybrid")
async def query_hybrid(request: Request):
    """
    Hybrid endpoint: Accept audio via POST, process it, and return a session ID for SSE streaming.
    """
    print(f"🔵 Hybrid endpoint called")
    try:
        if whisper_model is None:
            print("🔴 Whisper not available")
            return JSONResponse({"error": "Whisper not available"}, status_code=500)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        print(f"🔵 Created session: {session_id}")
        
        # Handle audio data from request body
        audio_data = None
        try:
            audio_data = await request.body()
            print(f"🔵 Audio data size: {len(audio_data)} bytes")
        except Exception as e:
            print(f"🔴 Error reading request body: {e}")
            return JSONResponse({"error": "Failed to read audio data"}, status_code=400)
        
        if not audio_data:
            print("🔴 No audio data received")
            return JSONResponse({"error": "No audio data provided"}, status_code=400)
        
        print(f"🔵 Starting background processing for session: {session_id}")
        
        # Process audio in background
        async def process_audio_background():
            audio_processing_start = time.time()  # Define this at the start
            temp_path = None
            converted_path = None
            
            try:
                print(f"🟡 Starting background processing for session: {session_id}")
                
                # Initialize session immediately to prevent "not found" errors
                session_store[session_id] = {
                    "status": "processing",
                    "transcript": "",
                    "query_text": "",
                    "contexts": [],
                    "response": "",
                    "chunks": []
                }
                print(f"🟢 Session initialized in store: {session_id}")
                
                temp_path, converted_path = process_audio_for_whisper(audio_data)
                query_text = transcribe_audio(temp_path, whisper_model, using_fast_whisper)
                
                if not query_text:
                    print(f"🔴 Transcription failed for session: {session_id}")
                    session_store[session_id] = {
                        "status": "error",
                        "error": "Transcription failed"
                    }
                    return
                
                print(f"🟢 Transcription completed for session: {session_id}")
                print(f"🟢 Transcript text: {query_text}")
                # Update session data
                session_store[session_id].update({
                    "status": "transcribed",
                    "transcript": query_text,
                    "query_text": query_text,
                    "audio_processing_latency": time.time() - audio_processing_start
                })
                
                # Retrieve context
                contexts = []
                search_results = None
                retrieval_start = time.time()
                if vector_store is not None:
                    try:
                        search_results = vector_store.search(query_text, n_results=3)
                        if search_results and search_results['documents']:
                            contexts = search_results['documents']
                        else:
                            contexts = ["Sample context for testing"]
                    except Exception as e:
                        print(f"Context retrieval error: {e}")
                        contexts = ["Sample context for testing"]
                else:
                    contexts = ["Sample context for testing"]
                
                retrieval_latency = time.time() - retrieval_start
                session_store[session_id]["contexts"] = contexts
                session_store[session_id]["search_results"] = search_results
                session_store[session_id]["status"] = "context_retrieved"
                session_store[session_id]["retrieval_latency"] = retrieval_latency
                print(f"🟢 Context retrieved for session: {session_id}")
                
                # Generate LLM response
                llm_start = time.time()
                if xai_client is not None and system is not None and user is not None:
                    try:
                        chat = xai_client.chat.create(model="grok-3-mini-fast", temperature=0)
                        system_prompt = f"You are a helpful AI assistant. Answer the question based on the provided context. Only give the response and directly answer the questions Context, keep the answer short: {str(contexts)}"
                        chat.append(system(system_prompt))
                        chat.append(user(str(query_text)))
                        
                        response_text = ""
                        chunk_count = 0
                        pending_chunks = []
                        
                        try:
                            for chunk in chat.stream():
                                chunk_count += 1
                                chunk_content = ""
                                
                                if isinstance(chunk, tuple) and len(chunk) > 1:
                                    chunk_data = chunk[1]
                                    if hasattr(chunk_data, 'content') and chunk_data.content:
                                        chunk_content = chunk_data.content
                                elif hasattr(chunk, 'content') and chunk.content:
                                    chunk_content = chunk.content
                                
                                if chunk_content:
                                    response_text += chunk_content
                                    # Store chunk for SSE streaming
                                    chunk_data = {
                                        'type': 'chunk',
                                        'content': chunk_content,
                                        'chunk_number': chunk_count
                                    }
                                    pending_chunks.append(chunk_data)
                                    print(f"🟢 Chunk {chunk_count} created: {chunk_content}")
                                    
                                    # Update session with new chunks
                                    session_store[session_id]["pending_chunks"] = pending_chunks
                                    session_store[session_id]["new_chunks_available"] = True
                                    
                                    # Small delay to make streaming visible
                                    await asyncio.sleep(0.005)
                        except Exception as stream_error:
                            print(f"Streaming failed, falling back to regular response: {stream_error}")
                            # Fallback to regular response
                            response = chat.sample()
                            response_text = response.content
                        
                        llm_latency = time.time() - llm_start
                        total_latency = time.time() - audio_processing_start
                        
                        session_store[session_id]["status"] = "complete"
                        session_store[session_id]["response"] = response_text
                        session_store[session_id]["llm_latency"] = llm_latency
                        session_store[session_id]["total_latency"] = total_latency
                        print(f"🟢 LLM response complete for session: {session_id}")
                        
                    except Exception as e:
                        print(f"LLM error: {e}")
                        session_store[session_id] = {
                            "status": "error",
                            "error": f"LLM error: {e}"
                        }
                else:
                    session_store[session_id] = {
                        "status": "error",
                        "error": "LLM backend not configured"
                    }
                
            except Exception as e:
                print(f"Background processing error: {e}")
                session_store[session_id] = {
                    "status": "error",
                    "error": f"Processing error: {e}"
                }
            finally:
                # Clean up temp files
                if temp_path or converted_path:
                    cleanup_paths = [path for path in [temp_path, converted_path] if path is not None]
                    if cleanup_paths:
                        AudioProcessor.cleanup_temp_files(*cleanup_paths)
        
        # Start background processing immediately
        asyncio.create_task(process_audio_background())
        
        # Return session ID immediately
        print(f"🟢 Returning session ID: {session_id}")
        return {"session_id": session_id, "status": "processing"}
        
    except Exception as e:
        print(f"🔴 Hybrid endpoint error: {e}")
        return JSONResponse({"error": f"Unexpected error: {e}"}, status_code=500)

@app.get("/stream/{session_id}")
async def stream_session(session_id: str):
    """
    SSE endpoint for streaming session data.
    """
    async def generate_sse():
        print(f"🔵 SSE connection started for session: {session_id}")
        # Wait for session to be created (max 2 seconds since we create it immediately now)
        wait_count = 0
        while session_id not in session_store and wait_count < 20:  # 20 * 0.1s = 2s
            await asyncio.sleep(0.1)
            wait_count += 1
        
        if session_id not in session_store:
            print(f"🔴 Session not found after waiting: {session_id}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found or timeout'})}\n\n"
            return
        
        print(f"🟢 Session found, starting stream: {session_id}")
        # Now stream the session data
        while True:
            if session_id not in session_store:
                print(f"🔴 Session expired during streaming: {session_id}")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session expired'})}\n\n"
                break
            
            session_data = session_store[session_id]
            print(f"🟡 Session status: {session_id} - {session_data['status']}")
            
            if session_data["status"] == "error":
                print(f"🔴 Session error: {session_id} - {session_data['error']}")
                yield f"data: {json.dumps({'type': 'error', 'message': session_data['error']})}\n\n"
                break
            
            elif session_data["status"] == "processing":
                # Session is being processed, send initial status
                if not session_data.get("processing_sent", False):
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Processing audio...'})}\n\n"
                    session_store[session_id]["processing_sent"] = True
            
            elif session_data["status"] == "transcribed":
                if not session_data.get("transcript_sent", False):
                    print(f"🟢 Sending transcript: {session_id}")
                    yield f"data: {json.dumps({'type': 'transcript', 'text': session_data['transcript']})}\n\n"
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
                    # Mark as sent to avoid duplicate
                    session_store[session_id]["transcript_sent"] = True
            
            elif session_data["status"] == "context_retrieved":
                # Always send transcript first if not already sent
                if not session_data.get("transcript_sent", False) and session_data.get("transcript"):
                    print(f"🟢 Sending transcript (delayed): {session_id}")
                    yield f"data: {json.dumps({'type': 'transcript', 'text': session_data['transcript']})}\n\n"
                    session_store[session_id]["transcript_sent"] = True
                
                if not session_data.get("context_sent", False):
                    print(f"🟢 Sending context retrieved status: {session_id}")
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
                    session_store[session_id]["context_sent"] = True
            
            elif session_data["status"] == "complete":
                print(f"🟢 Session complete, sending final chunks: {session_id}")
                
                # Always send transcript first if not already sent
                if not session_data.get("transcript_sent", False) and session_data.get("transcript"):
                    print(f"🟢 Sending transcript (final): {session_id}")
                    yield f"data: {json.dumps({'type': 'transcript', 'text': session_data['transcript']})}\n\n"
                    session_store[session_id]["transcript_sent"] = True
                
                # Send any remaining chunks that haven't been sent
                sent_chunks = session_data.get("sent_chunks", set())
                
                # Check for pending chunks first
                pending_chunks = session_data.get("pending_chunks", [])
                for chunk_data in pending_chunks:
                    chunk_id = f"{chunk_data['chunk_number']}_{chunk_data['content']}"
                    if chunk_id not in sent_chunks:
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        sent_chunks.add(chunk_id)
                        # Small delay to make streaming visible
                        await asyncio.sleep(0.005)
                
                # Send final response with latency stats and search results
                final_response = {
                    "type": "complete",
                    "full_response": session_data["response"],
                    "audio_processing_latency": session_data.get("audio_processing_latency", 0),
                    "retrieval_latency": session_data.get("retrieval_latency", 0),
                    "llm_latency": session_data.get("llm_latency", 0),
                    "total_latency": session_data.get("total_latency", 0)
                }
                
                # Add search results metadata if available
                search_results = session_data.get("search_results")
                if search_results and search_results.get('metadatas'):
                    retrieved_docs = []
                    for i, metadata in enumerate(search_results['metadatas']):
                        if i < 3:  # Limit to top 3 documents
                            doc_info = {
                                "source": metadata.get("source", "Unknown"),
                                "original_file": metadata.get("original_file", metadata.get("source", "Unknown")),
                                "similarity_score": search_results['distances'][i] if i < len(search_results['distances']) else 0
                            }
                            retrieved_docs.append(doc_info)
                    final_response["retrieved_documents"] = retrieved_docs
                yield f"data: {json.dumps(final_response)}\n\n"
                break
            
            # Check for new chunks during processing
            elif session_data["status"] == "context_retrieved" and session_data.get("new_chunks_available", False):
                # Send any new chunks that haven't been sent yet
                sent_chunks = session_data.get("sent_chunks", set())
                new_chunks_sent = False
                
                # Check for pending chunks
                pending_chunks = session_data.get("pending_chunks", [])
                for chunk_data in pending_chunks:
                    chunk_id = f"{chunk_data['chunk_number']}_{chunk_data['content']}"
                    if chunk_id not in sent_chunks:
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        sent_chunks.add(chunk_id)
                        new_chunks_sent = True
                        # Small delay to make streaming visible
                        await asyncio.sleep(0.005)
                
                if new_chunks_sent:
                    session_store[session_id]["sent_chunks"] = sent_chunks
                    session_store[session_id]["new_chunks_available"] = False
            
            await asyncio.sleep(0.005)  # Poll every 5ms for faster response
        
        # Clean up session after streaming
        if session_id in session_store:
            print(f"🧹 Cleaning up session: {session_id}")
            # Keep session alive for 5 seconds after completion to allow for multiple connections
            await asyncio.sleep(5)
            if session_id in session_store:
                del session_store[session_id]
                print(f"🧹 Session {session_id} finally cleaned up")
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if startup_complete else "starting up",
        "startup_complete": startup_complete,
        "whisper_loaded": whisper_model is not None,
        "whisper_type": "fast_whisper" if using_fast_whisper else "standard_whisper" if whisper_model is not None else "none",
        "qdrant_loaded": vector_store is not None,
        "xai_configured": xai_client is not None,
        "chunker_configured": chunker is not None,
        "chunking_config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "model_name": "grok-3-mini-fast"
        } if chunker else None,
        "vector_store_info": vector_store.get_info() if vector_store else None
    }
