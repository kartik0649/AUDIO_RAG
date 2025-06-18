import os
import chromadb
from chromadb.config import Settings

# Test basic ChromaDB functionality
print("Testing ChromaDB basic functionality...")

try:
    # Create a simple in-memory client
    client = chromadb.Client()
    print("ChromaDB client created successfully")
    
    # Create a collection without embedding function
    collection = client.create_collection("test_collection")
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