import os
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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
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

# Optional imports
try:
    import whisper
except ImportError:
    whisper = None

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
    global vector_store, whisper_model, startup_complete, chunker
    print("Starting up Audio RAG system with Qdrant...")
    
    # Initialize token-based chunker
    print(f"Initializing token-based chunker (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    try:
        chunker = create_chunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print("✅ Token-based chunker initialized successfully")
    except Exception as e:
        print(f"Error initializing chunker: {e}")
        chunker = None
    
    if whisper is not None:
        print("Loading Whisper model...")
        try:
            whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            whisper_model = None
    else:
        print("Warning: whisper not installed")
    
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
                    result = whisper_model.transcribe(whisper_path, fp16=False)
                    query_text = result.get("text", "")
                    if query_text and isinstance(query_text, str):
                        query_text = query_text.strip()
                        if query_text:
                            print(f"Original file transcription successful: {query_text}")
                            transcription_success = True
                        else:
                            print("Original file transcription returned empty text")
                    else:
                        print("Original file transcription returned invalid text")
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
                    result = whisper_model.transcribe(whisper_path, fp16=False)
                    query_text = result.get("text", "")
                    if query_text and isinstance(query_text, str):
                        query_text = query_text.strip()
                        if query_text:
                            print(f"Converted file transcription successful: {query_text}")
                            transcription_success = True
                        else:
                            print("Converted file transcription returned empty text")
                    else:
                        print("Converted file transcription returned invalid text")
                except Exception as e:
                    print(f"Converted file transcription failed: {e}")
            
            # Method 3: Try with different Whisper parameters
            if not transcription_success and temp_path:
                try:
                    print("Attempting transcription with adjusted parameters...")
                    result = whisper_model.transcribe(
                        temp_path, 
                        fp16=False, 
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
            
            # Method 4: Try with numpy array
            if not transcription_success and temp_path:
                try:
                    print("Attempting transcription with numpy array...")
                    import librosa
                    
                    # Load audio as numpy array
                    audio_array, sample_rate = librosa.load(temp_path, sr=16000)
                    
                    # Try to transcribe directly from numpy array
                    result = whisper_model.transcribe(audio_array, fp16=False)
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
                
                # Get the response using chat.sample()
                response = chat.sample()
                response_text = response.content  # This is a string, safe for JSON
                
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

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if startup_complete else "starting up",
        "startup_complete": startup_complete,
        "whisper_loaded": whisper_model is not None,
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
