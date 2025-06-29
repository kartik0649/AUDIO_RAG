#!/usr/bin/env python3
"""
Test script for token-based chunking functionality.
"""

import os
import sys
from text_chunker import create_chunker

def test_chunking():
    """Test the token-based chunking functionality."""
    
    print("🧪 Testing Token-Based Chunking")
    print("=" * 50)
    
    # Create chunker
    chunker = create_chunker(chunk_size=512, chunk_overlap=50)
    print(f"✅ Chunker created with chunk_size=512, overlap=50")
    
    # Test document
    test_content = """
    This is a test document for token-based chunking. It contains multiple paragraphs to test how the chunking algorithm works.

    The second paragraph discusses the importance of proper document chunking in RAG systems. Token-based chunking is more accurate than character-based chunking because it respects the actual semantic units that language models understand.

    The third paragraph explains that overlap between chunks helps maintain context and prevents important information from being lost at chunk boundaries. This is especially important for questions that span multiple chunks.

    The fourth paragraph demonstrates that the RecursiveCharacterTextSplitter from LangChain can intelligently split text at natural boundaries like paragraphs, sentences, and words, while respecting the token limit.

    The fifth paragraph shows that tiktoken provides accurate token counting that matches what OpenAI's models actually use, ensuring our chunks fit properly within context windows.

    The sixth paragraph concludes that this approach provides better retrieval performance and more accurate responses in RAG systems compared to simple character-based splitting.

    The seventh paragraph mentions that the chunking configuration can be easily adjusted through environment variables, making it flexible for different use cases and model requirements.

    The eighth paragraph notes that the system also tracks metadata about each chunk, including token counts and chunk indices, which helps with debugging and optimization.

    The ninth paragraph explains that this implementation integrates seamlessly with the existing FAISS vector store and maintains backward compatibility with the current system.

    The tenth paragraph summarizes that token-based chunking with overlap is a significant improvement that will enhance the quality of the RAG system's responses.
    """
    
    print(f"\n📄 Test document length: {len(test_content)} characters")
    
    # Get chunking info
    chunk_info = chunker.get_chunk_info(test_content)
    print(f"📊 Chunking info:")
    print(f"   - Total tokens: {chunk_info['total_tokens']}")
    print(f"   - Estimated chunks: {chunk_info['estimated_chunks']}")
    print(f"   - Chunk size: {chunk_info['chunk_size']}")
    print(f"   - Chunk overlap: {chunk_info['chunk_overlap']}")
    
    # Create chunks
    print(f"\n✂️  Creating chunks...")
    chunk_objects = chunker.chunk_document(test_content, {
        "source": "test_document.md",
        "original_file": "test_document.md"
    })
    
    print(f"✅ Created {len(chunk_objects)} chunks")
    
    # Display chunks
    for i, chunk_obj in enumerate(chunk_objects):
        content = chunk_obj["content"]
        metadata = chunk_obj["metadata"]
        
        print(f"\n📝 Chunk {i+1}/{len(chunk_objects)}:")
        print(f"   - Tokens: {metadata['token_count']}")
        print(f"   - Characters: {len(content)}")
        print(f"   - Source: {metadata['source']}")
        print(f"   - Preview: {content[:100]}...")
        
        # Verify token count
        actual_tokens = chunker._token_length_function(content)
        if actual_tokens != metadata['token_count']:
            print(f"   ⚠️  Token count mismatch: {actual_tokens} vs {metadata['token_count']}")
        else:
            print(f"   ✅ Token count verified: {actual_tokens}")
    
    print(f"\n🎯 Test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_chunking()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 