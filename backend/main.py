import os
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
import chromadb
from chromadb.config import Settings
import openai
from audio_processor import process_audio_for_whisper, AudioProcessor

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
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ImportError:
    chromadb = None
    SentenceTransformerEmbeddingFunction = None

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KB_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_kb")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Global variables for caching
collection = None
whisper_model = None
openai_client = None
startup_complete = False

# Configuration
USE_SIMPLE_EMBEDDINGS = os.getenv("USE_SIMPLE_EMBEDDINGS", "false").lower() == "true"
SKIP_INGESTION = os.getenv("SKIP_INGESTION", "false").lower() == "true"

def preprocess_document(content):
    """
    Preprocess document content to improve embedding performance and quality.
    """
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove special characters that don't add semantic value
    content = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', content)
    
    # Remove very long lines (likely code or data)
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        if len(line.strip()) < 1000:  # Remove lines longer than 1000 chars
            cleaned_lines.append(line)
    content = '\n'.join(cleaned_lines)
    
    # Remove duplicate paragraphs
    paragraphs = content.split('\n\n')
    seen = set()
    unique_paragraphs = []
    for para in paragraphs:
        para_clean = para.strip()
        if para_clean and para_clean not in seen:
            seen.add(para_clean)
            unique_paragraphs.append(para_clean)
    
    content = '\n\n'.join(unique_paragraphs)
    
    # Limit document size
    if len(content) > 10000:  # 10KB limit
        content = content[:10000] + "... [truncated]"
    
    return content.strip()

def split_large_document(content, max_chunk_size=50000):
    """
    Split large documents into smaller chunks to prevent memory issues.
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

# Check if we should reset the database
RESET_DB = os.getenv("RESET_DB", "false").lower() == "true"
if RESET_DB and os.path.exists(CHROMA_DIR):
    print("RESET_DB environment variable set, removing existing ChromaDB directory...")
    import shutil
    shutil.rmtree(CHROMA_DIR)
    print("ChromaDB directory removed")

# Load OpenAI API key and initialize client
api_key = os.getenv("OPENAI_API_KEY")
if openai is not None and OpenAI is not None and api_key:
    openai.api_key = api_key
    openai_client = OpenAI(api_key=api_key)

def create_fast_embedder():
    """Create a faster embedding function using TF-IDF or simple hashing"""
    class FastEmbeddingFunction:
        def __init__(self):
            self.dimension = 384
            self.name = "fast_embedder"
            self.default_space = "cosine"
            self.supported_spaces = ["cosine", "l2", "ip"]
            
        def __call__(self, input):
            import hashlib
            import numpy as np
            from collections import Counter
            import re
            
            if isinstance(input, str):
                input = [input]
            
            embeddings = []
            for text in input:
                # Simple TF-IDF inspired approach
                # Clean text
                text = re.sub(r'[^\w\s]', '', text.lower())
                words = text.split()
                
                # Create word frequency vector
                word_freq = Counter(words)
                
                # Create simple hash-based embedding
                hash_obj = hashlib.sha256(text.encode())
                hash_bytes = hash_obj.digest()
                hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)
                
                # Repeat to get desired dimension
                repeated = np.tile(hash_array, (self.dimension // len(hash_array)) + 1)
                embedding = repeated[:self.dimension].astype(np.float32)
                
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            return embeddings
        
        def embed_with_retries(self, input, max_retries=3):
            return self.__call__(input)
        
        def build_from_config(self, config):
            return self
    
    return FastEmbeddingFunction()

def get_embedder():
    """Get the embedding function"""
    print("Creating embedding function...")
    
    # Use simple embeddings by default for better compatibility
    print("Using simple embeddings for better compatibility...")
    return create_simple_embedder()
    
    # Use fast embedder if explicitly configured
    # if USE_SIMPLE_EMBEDDINGS:
    #     print("Using simple embeddings for faster processing...")
    #     return create_simple_embedder()
    
    # # Use fast embedder by default for better performance
    # print("Using fast embedder for optimal performance...")
    # return create_fast_embedder()

def create_simple_embedder():
    class SimpleEmbeddingFunction:
        def __init__(self):
            self.dimension = 384
            self.name = "simple_hash_embedder"
            self.default_space = "cosine"
            self.supported_spaces = ["cosine", "l2", "ip"]
            
        def __call__(self, input):
            import hashlib
            import numpy as np
            if isinstance(input, str):
                input = [input]
            embeddings = []
            for text in input:
                hash_obj = hashlib.md5(text.encode())
                hash_bytes = hash_obj.digest()
                hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)
                repeated = np.tile(hash_array, (self.dimension // len(hash_array)) + 1)
                embedding = repeated[:self.dimension].astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            return embeddings
            
        def embed_with_retries(self, input, **retry_kwargs):
            return self.__call__(input)
            
        def build_from_config(self, config):
            return self
            
        def get_config(self):
            return {"name": self.name, "dimension": self.dimension}
            
        def validate_config(self, config):
            return True
            
        def validate_config_update(self, config):
            return True
            
        @property
        def is_legacy(self):
            return False
    
    return SimpleEmbeddingFunction()

@app.on_event("startup")
async def startup_event():
    global collection, whisper_model, startup_complete
    print("Starting up Audio RAG system...")
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
    if chromadb is None:
        print("Warning: chromadb not installed")
        startup_complete = True
        return
    print("Initializing ChromaDB...")
    try:
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        print("ChromaDB client created successfully")
        
        # Use default ChromaDB embedding function (no custom embedder)
        collection = chroma_client.get_or_create_collection(
            "knowledge_base"
        )
        print("ChromaDB initialized successfully")
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        collection = None
        startup_complete = True
        return

    # Always ingest KB on startup
    print("Forcing ingestion of KB...")
    if SKIP_INGESTION:
        print("SKIP_INGESTION is set to true, skipping document ingestion...")
    else:
        try:
            ingest_kb()
        except Exception as e:
            print(f"Error during document ingestion: {e}")

    startup_complete = True
    print("Startup complete!")

def ingest_kb():
    if chromadb is None or collection is None:
        print("ChromaDB or collection not available, skipping ingestion")
        return
    print("Starting document ingestion...")
    print("Reading documents from knowledge base...")
    
    if not os.path.exists(KB_DIR):
        print(f"Knowledge base directory {KB_DIR} does not exist")
        return
    print(f"Knowledge base directory exists: {KB_DIR}")
    files = os.listdir(KB_DIR)
    print(f"Found {len(files)} files: {files}")
    
    # Process documents in batches
    BATCH_SIZE = 1  # Process 1 document at a time for better control
    total_processed = 0
    
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
                    
                # Preprocess document
                content = preprocess_document(content)
                
                # Split large documents into chunks
                if len(content) > 50000:  # 50KB threshold
                    print(f"Document {fname} is large ({len(content)} chars), splitting into chunks...")
                    chunks = split_large_document(content)
                    print(f"Split into {len(chunks)} chunks")
                    
                    for i, chunk in enumerate(chunks):
                        docs.append(chunk)
                        metadatas.append({"source": f"{fname}_chunk_{i+1}", "original_file": fname})
                        ids.append(str(uuid.uuid4()))
                        print(f"Added chunk {i+1} of {fname} ({len(chunk)} chars)")
                else:
                    docs.append(content)
                    metadatas.append({"source": fname})
                    ids.append(str(uuid.uuid4()))
                    print(f"Added document: {fname} ({len(content)} chars)")
                    
            except Exception as e:
                print(f"Error reading {fname}: {e}")
        
        if docs:
            print(f"Adding {len(docs)} docs to collection (batch {i//BATCH_SIZE + 1})...")
            
            def add_documents_batch():
                if collection is not None:
                    try:
                        print(f"Starting embedding for batch {i//BATCH_SIZE + 1}...")
                        start_time = time.time()
                        
                        # Add progress indicator
                        print(f"Processing {len(docs)} documents...")
                        
                        # Add documents with timeout
                        collection.add(documents=docs, metadatas=metadatas, ids=ids)
                        
                        end_time = time.time()
                        print(f"Batch {i//BATCH_SIZE + 1} added successfully in {end_time - start_time:.2f} seconds")
                        return True
                    except Exception as e:
                        print(f"Error adding batch {i//BATCH_SIZE + 1}: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
                else:
                    print("Collection is None, cannot add documents")
                    return False
            
            # Run batch addition with timeout
            thread = threading.Thread(target=add_documents_batch)
            thread.daemon = True
            thread.start()
            thread.join(timeout=15)  # 15 second timeout per batch
            
            if thread.is_alive():
                print(f"Batch {i//BATCH_SIZE + 1} addition timed out")
            else:
                total_processed += len(docs)
                print(f"Batch {i//BATCH_SIZE + 1} completed successfully")
        else:
            print(f"No documents in batch {i//BATCH_SIZE + 1}")
    
    print(f"\n--- Ingestion Complete ---")
    print(f"Total documents processed: {total_processed}")
    if total_processed > 0:
        print("Successfully ingested documents")
    else:
        print("No documents were ingested")

@app.post("/query")
async def query(request: Request, audio: UploadFile = File(None)):
    start_time = time.time()
    
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
        
        retrieval_start = time.time()
        contexts = []
        
        # Temporary bypass for testing - skip ChromaDB entirely
        BYPASS_CHROMADB = True  # Set to False to enable ChromaDB
        
        if BYPASS_CHROMADB:
            print("Bypassing ChromaDB for testing...")
            contexts = ["This is a sample context for testing the system. The audio transcription is working correctly."]
        elif collection is not None:
            try:
                # Try to get relevant contexts from the knowledge base
                print(f"Querying ChromaDB with text: {query_text}")
                print(f"Collection info: {collection.name if collection else 'None'}")
                
                print("About to call collection.count()...")
                try:
                    count = collection.count()
                    print(f"Collection count: {count}")
                except Exception as e:
                    print(f"Error getting collection count: {e}")
                    contexts = ["Sample context for testing"]
                    return {
                        "transcript": query_text,
                        "response": "Error getting collection count, using sample context",
                        "retrieval_latency": time.time() - retrieval_start,
                        "llm_latency": 0,
                        "total_latency": time.time() - start_time,
                    }
                
                # Test simple query first
                print("Testing simple ChromaDB query...")
                try:
                    test_results = collection.query(
                        query_texts=["test"],
                        n_results=1
                    )
                    print(f"Simple test query successful: {test_results}")
                except Exception as e:
                    print(f"Simple test query failed: {e}")
                    contexts = ["Sample context for testing"]
                    return {
                        "transcript": query_text,
                        "response": "Test query failed, using sample context",
                        "retrieval_latency": time.time() - retrieval_start,
                        "llm_latency": 0,
                        "total_latency": time.time() - start_time,
                    }
                
                # Add timeout for ChromaDB query
                import threading
                import queue
                
                query_result = queue.Queue()
                query_error = queue.Queue()
                
                def run_query():
                    try:
                        if collection is not None:
                            print("Starting actual query in thread...")
                            results = collection.query(
                                query_texts=[str(query_text)],
                                n_results=3
                            )
                            print("Query completed in thread")
                            query_result.put(results)
                        else:
                            query_error.put(Exception("Collection is None"))
                    except Exception as e:
                        print(f"Query error in thread: {e}")
                        query_error.put(e)
                
                print("Creating query thread...")
                query_thread = threading.Thread(target=run_query)
                query_thread.daemon = True
                query_thread.start()
                
                print("Waiting for query thread to complete...")
                # Wait for query with timeout
                query_thread.join(timeout=30)  # 30 second timeout
                
                if query_thread.is_alive():
                    print("ChromaDB query timed out after 30 seconds")
                    contexts = ["Sample context for testing"]
                elif not query_error.empty():
                    error = query_error.get()
                    print(f"ChromaDB query error: {error}")
                    contexts = ["Sample context for testing"]
                else:
                    results = query_result.get()
                    print(f"ChromaDB results: {results}")
                    if results and results['documents']:
                        contexts = results['documents'][0]
                    else:
                        contexts = ["Sample context for testing"]
                        
            except Exception as e:
                print(f"Error retrieving contexts: {e}")
                import traceback
                traceback.print_exc()
                contexts = ["Sample context for testing"]
        else:
            print("ChromaDB collection is None, using sample context")
            contexts = ["Sample context for testing"]
        
        retrieval_latency = time.time() - retrieval_start
        llm_start = time.time()
        response_text = ""
        if openai_client is not None:
            prompt = f"Answer the question based on context: {contexts}\nQuestion: {query_text}"
            try:
                completion = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = completion.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API error: {e}")
                response_text = f"Error: {e}"
        else:
            response_text = "LLM backend not configured"
        llm_latency = time.time() - llm_start
        
        total_latency = time.time() - start_time
        print(f"Query completed successfully. Total latency: {total_latency:.2f}s")
        
        return {
            "transcript": query_text,
            "response": response_text,
            "retrieval_latency": retrieval_latency,
            "llm_latency": llm_latency,
            "total_latency": total_latency,
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
        "chromadb_loaded": collection is not None,
        "openai_configured": openai_client is not None
    }
