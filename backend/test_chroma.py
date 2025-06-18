import os
import chromadb
from chromadb.config import Settings

# Test basic ChromaDB functionality
print("Testing ChromaDB basic functionality...")

try:
    # Create a simple in-memory client
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    print("ChromaDB client created successfully")
    
    # Create a collection without embedding function
    # Use a simple embedding function to avoid network downloads
    class DummyEmbedder:
        """Simple embedder that returns zero vectors"""

        def __call__(self, input):
            if isinstance(input, str):
                input = [input]
            return [[0.0] * 384 for _ in input]

        # Required methods for ChromaDB embedding function interface
        def embed_with_retries(self, texts, max_retries=3):
            return self.__call__(texts)

        def build_from_config(self, config):
            return self

    collection = client.create_collection(
        "test_collection",
        embedding_function=DummyEmbedder(),
    )
    print("Collection created successfully")
    
    # Add a simple document
    collection.add(
        documents=["This is a test document"],
        metadatas=[{"source": "test"}],
        ids=["test_id_1"]
    )
    print("Document added successfully")
    
    # Query the collection
    results = collection.query(query_texts=["test"], n_results=1)
    print(f"Query results: {results}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 