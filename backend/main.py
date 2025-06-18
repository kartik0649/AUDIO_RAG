from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uuid
import time
import os
import typing
import threading
import signal

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

def get_embedder():
    """Lazy load the embedding function only when needed with timeout"""
    print("Creating embedding function...")
    if SentenceTransformerEmbeddingFunction is not None:
        try:
            # Use a timeout to prevent hanging
            embedder_result: list[typing.Any] = [None]
            embedder_error: list[typing.Any] = [None]
            
            def create_embedder():
                try:
                    # Use a smaller, faster model
                    if SentenceTransformerEmbeddingFunction is not None:
                        embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                        embedder_result[0] = embedder
                except Exception as e:
                    embedder_error[0] = e
            
            # Create embedder in a separate thread with timeout
            thread = threading.Thread(target=create_embedder)
            thread.daemon = True
            thread.start()
            # Reduce the timeout so startup doesn't hang for long periods
            thread.join(timeout=10)
            
            if thread.is_alive():
                print("Embedding function creation timed out, using simple embeddings")
                return create_simple_embedder()
            elif embedder_error[0]:
                print(f"Error creating embedding function: {embedder_error[0]}")
                print("Using simple embeddings...")
                return create_simple_embedder()
            else:
                print("Embedding function created successfully")
                return embedder_result[0]
        except Exception as e:
            print(f"Error creating embedding function: {e}")
            print("Using simple embeddings...")
            return create_simple_embedder()
    else:
        print("SentenceTransformerEmbeddingFunction not available, using simple embeddings")
        return create_simple_embedder()

def create_simple_embedder():
    """Create a simple embedding function that doesn't require external dependencies"""
    class SimpleEmbeddingFunction:
        def __init__(self):
            self.dimension = 384  # Standard embedding dimension
            self.name = "simple_hash_embedder"
            self.default_space = "cosine"
            self.supported_spaces = ["cosine", "l2", "ip"]
            
        def __call__(self, input):
            # Simple hash-based embedding for demonstration
            # In production, you'd want a proper embedding model
            import hashlib
            import numpy as np
            
            if isinstance(input, str):
                input = [input]
            
            embeddings = []
            for text in input:
                # Create a simple hash-based embedding
                hash_obj = hashlib.md5(text.encode())
                hash_bytes = hash_obj.digest()
                
                # Convert hash to numpy array and pad/truncate to desired dimension
                hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)
                # Repeat the hash to get enough values
                repeated = np.tile(hash_array, (self.dimension // len(hash_array)) + 1)
                embedding = repeated[:self.dimension].astype(np.float32)
                
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            return embeddings
        
        def embed_with_retries(self, input, max_retries=3):
            """Required method for ChromaDB embedding function interface"""
            return self.__call__(input)
        
        def build_from_config(self, config):
            """Required method for ChromaDB embedding function interface"""
            return self
    
    return SimpleEmbeddingFunction()

@app.on_event("startup")
async def startup_event():
    global collection, whisper_model, startup_complete
    
    print("Starting up Audio RAG system...")
    
    # Load Whisper model once at startup
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
    
    # Initialize ChromaDB
    if chromadb is None:
        print("Warning: chromadb not installed")
        startup_complete = True
        return

    print("Initializing ChromaDB...")
    try:
        # Use persistent client for better reliability and disable telemetry
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Create or get collection - always use an embedding function
        embedder = get_embedder()
        collection = chroma_client.get_or_create_collection("knowledge_base", embedding_function=typing.cast(typing.Any, embedder))
        print("ChromaDB initialized successfully")
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        collection = None
        startup_complete = True
        return

    # Ingest knowledge base documents
    print("Ingesting knowledge base documents...")
    try:
        ingest_kb()
    except Exception as e:
        print(f"Error during document ingestion: {e}")
        import traceback
        traceback.print_exc()
    
    startup_complete = True
    print("Startup complete!")

def ingest_kb():
    if chromadb is None or collection is None:
        print("ChromaDB or collection not available, skipping ingestion")
        return
    
    print("Starting document ingestion...")
    
    # Check if documents already exist in collection with timeout
    try:
        print("Checking if documents already exist in collection...")
        if collection is not None:
            # Add timeout for count operation
            count_result: list[typing.Any] = [None]
            count_error: list[typing.Any] = [None]
            count: int = 0  # Initialize count variable
            
            def get_count():
                try:
                    if collection is not None:  # Double check for type safety
                        count_result[0] = collection.count()
                except Exception as e:
                    count_error[0] = e
            
            thread = threading.Thread(target=get_count)
            thread.daemon = True
            thread.start()
            thread.join(timeout=10)  # 10 second timeout for count operation
            
            if thread.is_alive():
                print("Collection count operation timed out, assuming empty collection")
                count = 0
            elif count_error[0]:
                print(f"Error getting collection count: {count_error[0]}")
                print("Assuming empty collection and proceeding with ingestion")
                count = 0
            else:
                count = count_result[0] if count_result[0] is not None else 0
            
            print(f"Current collection count: {count}")
            if count > 0:
                print(f"Collection already has {count} documents, skipping ingestion")
                return
    except Exception as e:
        print(f"Error checking collection count: {e}")
        print("Proceeding with ingestion anyway...")
        import traceback
        traceback.print_exc()
    
    print("Reading documents from knowledge base...")
    docs, metadatas, ids = [], [], []
    
    if not os.path.exists(KB_DIR):
        print(f"Knowledge base directory {KB_DIR} does not exist")
        return
    
    print(f"Knowledge base directory exists: {KB_DIR}")
    files = os.listdir(KB_DIR)
    print(f"Found {len(files)} files in knowledge base directory: {files}")
        
    for fname in files:
        path = os.path.join(KB_DIR, fname)
        print(f"Reading file: {fname} at path: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(content)
                metadatas.append({"source": fname})
                ids.append(str(uuid.uuid4()))
                print(f"Added document: {fname} (length: {len(content)} chars)")
        except Exception as e:
            print(f"Error reading file {fname}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Processed {len(docs)} documents")
    
    if docs and collection is not None:
        print(f"Attempting to add {len(docs)} documents to collection...")
        try:
            print("About to add documents to collection...")
            print(f"Document contents: {docs}")
            print(f"Metadata: {metadatas}")
            print(f"IDs: {ids}")
            
            # Add timeout for document addition
            def add_documents():
                if collection is not None:
                    print("Inside add_documents function, calling collection.add...")
                    collection.add(documents=docs, metadatas=metadatas, ids=ids)
                    print("collection.add completed successfully")
            
            thread = threading.Thread(target=add_documents)
            thread.daemon = True
            thread.start()
            print("Started document addition thread")
            thread.join(timeout=60)  # 60 second timeout for document addition
            
            if thread.is_alive():
                print("Document addition timed out")
                return
            else:
                print("Documents added to collection!")
                print(f"Successfully ingested {len(docs)} documents")
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No documents found to ingest or collection is None")

@app.post("/query")
async def query(audio: UploadFile = File(...)):
    """Main RAG pipeline: audio -> text -> retrieval -> LLM"""
    start_time = time.time()
    
    if whisper_model is None:
        return JSONResponse({"error": "whisper not available"}, status_code=500)

    # Save uploaded audio to temp file
    tmp_path = f"/tmp/{uuid.uuid4()}.wav"
    with open(tmp_path, "wb") as f:
        f.write(await audio.read())

    # Transcribe using pre-loaded model
    result = whisper_model.transcribe(tmp_path)
    os.remove(tmp_path)
    query_text = result.get("text", "")

    # Retrieve context
    retrieval_start = time.time()
    contexts = []
    if collection is not None:
        # Chroma expects a list of query strings
        results = collection.query(query_texts=[query_text], n_results=2)
        documents = results.get("documents", [])
        contexts = documents[0] if documents else []
    retrieval_latency = time.time() - retrieval_start

    # LLM response via OpenAI Chat (if available)
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
            response_text = f"Error: {str(e)}"
    else:
        response_text = "LLM backend not configured"
    llm_latency = time.time() - llm_start

    return {
        "transcript": query_text,
        "response": response_text,
        "retrieval_latency": retrieval_latency,
        "llm_latency": llm_latency,
        "total_latency": time.time() - start_time,
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if startup_complete else "starting up",
        "startup_complete": startup_complete,
        "whisper_loaded": whisper_model is not None,
        "chromadb_loaded": collection is not None,
        "openai_configured": openai_client is not None
    }
