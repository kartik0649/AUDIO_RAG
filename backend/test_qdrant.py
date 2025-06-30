#!/usr/bin/env python3
"""
Test script for Qdrant vector store implementation.
"""

import os
import sys
import tempfile
import shutil
from qdrant_vector_store import QdrantVectorStore

def test_qdrant_vector_store():
    """Test the Qdrant vector store functionality."""
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Initialize vector store
        print("Initializing Qdrant vector store...")
        vector_store = QdrantVectorStore(
            storage_path=temp_dir,
            use_persistent_storage=True,
            collection_name="test_collection"
        )
        
        # Test adding documents
        print("\nTesting document addition...")
        test_documents = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI that focuses on algorithms.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language."
        ]
        
        test_metadatas = [
            {"source": "test1", "original_file": "test1.txt"},
            {"source": "test2", "original_file": "test2.txt"},
            {"source": "test3", "original_file": "test3.txt"},
            {"source": "test4", "original_file": "test4.txt"}
        ]
        
        vector_store.add_documents(test_documents, test_metadatas)
        print(f"✅ Added {len(test_documents)} documents")
        
        # Test counting documents
        print(f"\nDocument count: {vector_store.count()}")
        
        # Test search functionality
        print("\nTesting search functionality...")
        query = "What is machine learning?"
        results = vector_store.search(query, n_results=2)
        
        print(f"Query: {query}")
        print(f"Found {len(results['documents'])} results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'], 
            results['metadatas'], 
            results['distances']
        )):
            print(f"  {i+1}. Distance: {distance:.4f}")
            print(f"     Document: {doc[:100]}...")
            print(f"     Source: {metadata.get('source', 'unknown')}")
        
        # Test vector store info
        print("\nVector store info:")
        info = vector_store.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test filtered search
        print("\nTesting filtered search...")
        filter_condition = {"source": "test2"}
        filtered_results = vector_store.search_with_filter(
            query, filter_condition, n_results=1
        )
        
        print(f"Filtered search results: {len(filtered_results['documents'])}")
        for doc in filtered_results['documents']:
            print(f"  - {doc[:100]}...")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")
    
    return True

if __name__ == "__main__":
    print("Testing Qdrant Vector Store Implementation")
    print("=" * 50)
    
    success = test_qdrant_vector_store()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1) 