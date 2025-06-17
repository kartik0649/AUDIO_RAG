# Prototype Voice Agent Report

This report gives a very high-level comparison of several backend options for
transcribing audio, performing retrieval augmented generation, and producing a
response.

## Latency

The prototype measures three latencies:

- **Retrieval latency** – time to query ChromaDB for relevant documents.
- **LLM latency** – time spent waiting for the language model response.
- **Total latency** – overall time from audio upload to final text response.

Exact numbers depend on hardware and model choice. Local small models will be
slower but cost less, while hosted APIs are typically faster but incur per-call
charges.

## Quality / Accuracy

- Whisper provides good transcription accuracy for English speech.
- Retrieval quality depends on the relevance of documents in ChromaDB and the
  embedding model used.
- Using larger language models usually yields more coherent answers but costs
  more and may have higher latency.

## Cost Estimates

- **OpenAI GPT-3.5**: priced per token (see OpenAI pricing). Each request may
  cost a fraction of a cent depending on prompt and response length.
- **Local models**: once downloaded, no per-call cost beyond compute time.
- **ChromaDB**: open-source and free to run locally.

## Architecture Justification

- **FastAPI** offers a lightweight Python backend with async support.
- **ChromaDB** is used for the vector store due to its easy setup and local
  persistence.
- **Expo/React Native** allows quick iteration of a cross-platform mobile UI for
  capturing audio.

Scaling this prototype would require moving the vector database and model
inference to managed services or more robust infrastructure. Caching of frequent
queries and batching can help reduce costs.
