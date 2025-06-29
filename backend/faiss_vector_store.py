import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    A simple FAISS-based vector store for document retrieval.
    Uses sentence-transformers for embeddings and FAISS for similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """
        Initialize the FAISS vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use
            dimension: Dimension of the embeddings
        """
        self.model_name = model_name
        self.dimension = dimension
        self.embedder = None
        self.index = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.is_trained = False
        
        # Initialize the embedding model
        self._init_embedder()
    
    def _init_embedder(self):
        """Initialize the sentence transformer embedder."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            raise
    
    def _create_index(self, num_vectors: int) -> faiss.Index:
        """
        Create a FAISS index for the given number of vectors.
        
        Args:
            num_vectors: Number of vectors to index
            
        Returns:
            FAISS index
        """
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(self.dimension)
        
        # For larger datasets, we could use IndexIVFFlat for better performance
        # if num_vectors > 1000:
        #     quantizer = faiss.IndexFlatIP(self.dimension)
        #     index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, num_vectors // 10))
        
        return index
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, 
                     ids: Optional[List[str]] = None) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Generate metadata if not provided
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]
        
        # Generate embeddings
        try:
            logger.info("Generating embeddings...")
            embeddings = self.embedder.encode(documents, show_progress_bar=True, batch_size=32)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to existing data or create new
        if self.index is None:
            # First time adding documents
            self.index = self._create_index(len(documents))
            self.documents = documents
            self.metadatas = metadatas
            self.ids = ids
        else:
            # Adding to existing index
            self.documents.extend(documents)
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)
        
        # Add vectors to index
        try:
            logger.info("Adding vectors to FAISS index...")
            self.index.add(embeddings.astype(np.float32))
            logger.info(f"Successfully added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS index: {e}")
            raise
    
    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("No documents in vector store")
            return {
                "documents": [],
                "metadatas": [],
                "ids": [],
                "distances": []
            }
        
        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query[:100]}...")
            query_embedding = self.embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            logger.info(f"Searching for {n_results} similar documents...")
            distances, indices = self.index.search(
                query_embedding.astype(np.float32), 
                min(n_results, len(self.documents))
            )
            
            # Prepare results
            results = {
                "documents": [self.documents[i] for i in indices[0] if i < len(self.documents)],
                "metadatas": [self.metadatas[i] for i in indices[0] if i < len(self.metadatas)],
                "ids": [self.ids[i] for i in indices[0] if i < len(self.ids)],
                "distances": distances[0].tolist()
            }
            
            logger.info(f"Found {len(results['documents'])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "ids": [],
                "distances": []
            }
    
    def count(self) -> int:
        """Get the number of documents in the vector store."""
        return len(self.documents) if self.documents else 0
    
    def save(self, filepath: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            filepath: Path to save the vector store
        """
        try:
            logger.info(f"Saving vector store to {filepath}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save FAISS index
            index_path = f"{filepath}.index"
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = f"{filepath}.metadata"
            metadata = {
                "documents": self.documents,
                "metadatas": self.metadatas,
                "ids": self.ids,
                "model_name": self.model_name,
                "dimension": self.dimension
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info("Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load(self, filepath: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            filepath: Path to load the vector store from
        """
        try:
            logger.info(f"Loading vector store from {filepath}")
            
            # Load FAISS index
            index_path = f"{filepath}.index"
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            else:
                logger.warning(f"Index file not found: {index_path}")
                return
            
            # Load metadata
            metadata_path = f"{filepath}.metadata"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.documents = metadata.get("documents", [])
                self.metadatas = metadata.get("metadatas", [])
                self.ids = metadata.get("ids", [])
                self.model_name = metadata.get("model_name", self.model_name)
                self.dimension = metadata.get("dimension", self.dimension)
                
                # Reinitialize embedder if needed
                if self.embedder is None:
                    self._init_embedder()
                
                logger.info(f"Loaded {len(self.documents)} documents from vector store")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.index = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        logger.info("Vector store cleared")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        return {
            "num_documents": len(self.documents),
            "model_name": self.model_name,
            "dimension": self.dimension,
            "has_index": self.index is not None
        } 