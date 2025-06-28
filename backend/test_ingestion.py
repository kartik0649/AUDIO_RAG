#!/usr/bin/env python3
"""
Test script for the improved ingestion process
"""

import os
import sys
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ingest_kb, preprocess_document, split_large_document

def test_preprocessing():
    """Test document preprocessing"""
    print("Testing document preprocessing...")
    
    # Test document
    test_content = """
    This is a test document with    lots   of   whitespace.
    
    It has duplicate paragraphs.
    
    It has duplicate paragraphs.
    
    And some very long lines that should be removed: """ + "x" * 2000 + """
    
    And some special characters: @#$%^&*()_+{}|:"<>?[]\\;'/,.
    """
    
    processed = preprocess_document(test_content)
    print(f"Original length: {len(test_content)}")
    print(f"Processed length: {len(processed)}")
    print(f"Processed content preview: {processed[:200]}...")
    
    return len(processed) < len(test_content)

def test_chunking():
    """Test document chunking"""
    print("\nTesting document chunking...")
    
    # Create a large test document
    large_content = "This is a test paragraph. " * 10000  # ~250KB
    
    chunks = split_large_document(large_content, max_chunk_size=50000)
    print(f"Original length: {len(large_content)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    
    return len(chunks) > 1

def test_ingestion():
    """Test the ingestion process"""
    print("\nTesting ingestion process...")
    
    try:
        start_time = time.time()
        ingest_kb()
        end_time = time.time()
        
        print(f"Ingestion completed in {end_time - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Improved Ingestion Process ===\n")
    
    # Test preprocessing
    preprocessing_ok = test_preprocessing()
    print(f"Preprocessing test: {'PASSED' if preprocessing_ok else 'FAILED'}")
    
    # Test chunking
    chunking_ok = test_chunking()
    print(f"Chunking test: {'PASSED' if chunking_ok else 'FAILED'}")
    
    # Test ingestion (only if not skipping)
    if not os.getenv("SKIP_INGESTION", "false").lower() == "true":
        ingestion_ok = test_ingestion()
        print(f"Ingestion test: {'PASSED' if ingestion_ok else 'FAILED'}")
    else:
        print("Skipping ingestion test (SKIP_INGESTION=true)")
    
    print("\n=== Test Complete ===") 