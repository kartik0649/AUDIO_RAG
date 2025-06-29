import os
import re
from typing import List, Dict, Any
import tiktoken

class TokenBasedChunker:
    """
    Token-based document chunking with overlap using tiktoken.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the token-based chunker.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            model_name: OpenAI model name for tokenizer
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Initialize tiktoken tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _token_length_function(self, text: str) -> int:
        """
        Calculate the number of tokens in a text string.
        
        Args:
            text: Input text string
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def _split_text_by_tokens(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count with overlap.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Calculate end position for this chunk
            end = start + self.chunk_size
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start:end]
            
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Clean up the chunk text (remove partial words at boundaries)
            chunk_text = self._clean_chunk_boundaries(chunk_text, text)
            
            chunks.append(chunk_text)
            
            # Move start position for next chunk (with overlap)
            start = end - self.chunk_overlap
            
            # If we're at the end, break
            if start >= len(tokens):
                break
        
        return chunks
    
    def _clean_chunk_boundaries(self, chunk_text: str, original_text: str) -> str:
        """
        Clean up chunk boundaries to avoid cutting words in the middle.
        
        Args:
            chunk_text: The chunk text to clean
            original_text: The original full text for reference
            
        Returns:
            Cleaned chunk text
        """
        # Remove leading/trailing whitespace
        chunk_text = chunk_text.strip()
        
        # Try to find complete sentences
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
        
        if len(sentences) > 1:
            # Keep all complete sentences except the last one if it's incomplete
            last_sentence = sentences[-1]
            if not last_sentence.endswith(('.', '!', '?')):
                # Remove incomplete last sentence
                chunk_text = '. '.join(sentences[:-1])
                if chunk_text:
                    chunk_text += '.'
        
        return chunk_text
    
    def chunk_document(self, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split a document into chunks based on token count.
        
        Args:
            content: Document content to chunk
            metadata: Optional metadata for the document
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not content or not content.strip():
            return []
        
        # Preprocess content
        content = self._preprocess_content(content)
        
        # Split into chunks
        chunks = self._split_text_by_tokens(content)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata is not None else {}
            chunk_metadata.update({
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "token_count": self._token_length_function(chunk),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            })
            
            chunk_objects.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return chunk_objects
    
    def _preprocess_content(self, content: str) -> str:
        """
        Preprocess document content before chunking.
        
        Args:
            content: Raw document content
            
        Returns:
            Preprocessed content
        """
        # Remove excessive whitespace but preserve structure
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Remove excessive blank lines
        content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces and tabs
        
        # Remove very long lines (likely code or data)
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            if len(line.strip()) < 2000:  # Increased limit for markdown content
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
        
        return content.strip()
    
    def get_chunk_info(self, content: str) -> Dict[str, Any]:
        """
        Get information about how a document would be chunked.
        
        Args:
            content: Document content
            
        Returns:
            Dictionary with chunking information
        """
        if not content or not content.strip():
            return {
                "total_tokens": 0,
                "estimated_chunks": 0,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        
        total_tokens = self._token_length_function(content)
        estimated_chunks = max(1, (total_tokens - self.chunk_overlap) // (self.chunk_size - self.chunk_overlap))
        
        return {
            "total_tokens": total_tokens,
            "estimated_chunks": estimated_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "model_name": self.model_name
        }

def create_chunker(chunk_size: int = 512, chunk_overlap: int = 50, model_name: str = "gpt-3.5-turbo") -> TokenBasedChunker:
    """
    Factory function to create a token-based chunker.
    
    Args:
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        model_name: OpenAI model name for tokenizer
        
    Returns:
        TokenBasedChunker instance
    """
    return TokenBasedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, model_name=model_name) 