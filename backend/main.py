from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uuid
import time
import os

try:
    import whisper
except ImportError:
    whisper = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    import openai
except ImportError:
    openai = None

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

client = None
collection = None

@app.on_event("startup")
def startup_event():
    global client, collection
    if chromadb is None:
        print("chromadb not installed")
        return
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    collection = client.get_or_create_collection("kb")
    ingest_kb()


def ingest_kb():
    if chromadb is None:
        return
    docs = []
    metadatas = []
    ids = []
    for fname in os.listdir(KB_DIR):
        with open(os.path.join(KB_DIR, fname), "r") as f:
            docs.append(f.read())
            metadatas.append({"source": fname})
            ids.append(str(uuid.uuid4()))
    if docs:
        collection.add(documents=docs, metadatas=metadatas, ids=ids)


@app.post("/query")
async def query(audio: UploadFile = File(...)):
    """Main RAG pipeline: audio -> text -> retrieval -> LLM"""
    start_time = time.time()
    if whisper is None:
        return JSONResponse({"error": "whisper not installed"}, status_code=500)
    # Save uploaded file
    tmp_path = f"/tmp/{uuid.uuid4()}.wav"
    with open(tmp_path, "wb") as f:
        f.write(await audio.read())
    model = whisper.load_model("base")
    result = model.transcribe(tmp_path)
    os.remove(tmp_path)
    query_text = result.get("text", "")

    retrieval_start = time.time()
    if collection is not None:
        results = collection.query(query_texts=[query_text], n_results=2)
        contexts = [doc for doc in results.get("documents", [[]])[0]]
    else:
        contexts = []
    retrieval_latency = time.time() - retrieval_start

    llm_start = time.time()
    response_text = ""
    if openai is not None:
        prompt = f"Answer the question based on context: {contexts}\nQuestion: {query_text}"
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        response_text = completion.choices[0].message.content
    else:
        response_text = "LLM backend not configured"
    llm_latency = time.time() - llm_start

    total_latency = time.time() - start_time
    return {
        "transcript": query_text,
        "response": response_text,
        "retrieval_latency": retrieval_latency,
        "llm_latency": llm_latency,
        "total_latency": total_latency,
    }
