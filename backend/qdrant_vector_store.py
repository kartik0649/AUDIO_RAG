import os
import json
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
    CreateCollection, CollectionInfo
)

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """
    A Qdrant-based vector store for document retrieval with persistent storage.
    Uses sentence-transformers for embeddings and Qdrant for similarity search.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 dimension: int = 384,
                 collection_name: str = "audio_rag_documents",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 use_persistent_storage: bool = True,
                 storage_path: str = "./qdrant_storage"):
        """
        Initialize the Qdrant vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use
            dimension: Dimension of the embeddings
            collection_name: Name of the Qdrant collection
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            use_persistent_storage: Whether to use persistent storage
            storage_path: Path for persistent storage (if use_persistent_storage is True)
        """
        self.model_name = model_name
        self.dimension = dimension
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.embedder = None
        self.client = None
        self.use_persistent_storage = use_persistent_storage
        self.storage_path = storage_path
        
        # Initialize the embedding model
        self._init_embedder()
        
        # Initialize Qdrant client
        self._init_qdrant_client()
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _init_embedder(self):
        """Initialize the sentence transformer embedder."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            raise
    
    def _init_qdrant_client(self):
        """Initialize the Qdrant client."""
        try:
            if self.use_persistent_storage:
                # Use local persistent storage
                logger.info(f"Initializing Qdrant client with persistent storage at {self.storage_path}")
                self.client = QdrantClient(path=self.storage_path)
            else:
                # Use remote Qdrant server
                logger.info(f"Initializing Qdrant client connecting to {self.qdrant_host}:{self.qdrant_port}")
                self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            
            logger.info("Qdrant client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't."""
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized")
                return
                
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                
                # Create payload indexes for better filtering performance
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source",
                    field_schema="keyword"
                )
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="original_file",
                    field_schema="keyword"
                )
                
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
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
        
        logger.info(f"Adding {len(documents)} documents to Qdrant vector store")
        
        # Generate IDs if not provided
        if ids is None:
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
        
        # Prepare points for Qdrant
        points = []
        for i, (doc_id, doc_text, metadata, embedding) in enumerate(zip(ids, documents, metadatas, embeddings)):
            point = PointStruct(
                id=doc_id,
                vector=embedding.tolist(),
                payload={
                    "text": doc_text,
                    "source": metadata.get("source", "unknown"),
                    "original_file": metadata.get("original_file", metadata.get("source", "unknown")),
                    "metadata": json.dumps(metadata)
                }
            )
            points.append(point)
        
        # Add points to collection
        try:
            logger.info("Adding points to Qdrant collection...")
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding points to Qdrant collection: {e}")
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
        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query[:100]}...")
            query_embedding = self.embedder.encode([query])
            
            # Search in Qdrant
            logger.info(f"Searching for {n_results} similar documents...")
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding[0].tolist(),
                limit=n_results,
                with_payload=True
            )
            
            # Prepare results
            documents = []
            metadatas = []
            ids = []
            distances = []
            
            for result in search_results:
                documents.append(result.payload.get("text", ""))
                metadata = json.loads(result.payload.get("metadata", "{}"))
                metadatas.append(metadata)
                ids.append(result.id)
                distances.append(1 - result.score)  # Convert similarity to distance
            
            results = {
                "documents": documents,
                "metadatas": metadatas,
                "ids": ids,
                "distances": distances
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
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def save(self, filepath: str) -> None:
        """
        Save the vector store to disk.
        Note: Qdrant handles persistence automatically when using persistent storage.
        This method is kept for compatibility but doesn't need to do anything.
        
        Args:
            filepath: Path to save the vector store (not used for Qdrant)
        """
        try:
            logger.info("Qdrant vector store is automatically persisted")
            # Qdrant handles persistence automatically when using persistent storage
            # No additional save operation needed
        except Exception as e:
            logger.error(f"Error in save operation: {e}")
            raise
    
    def load(self, filepath: str) -> None:
        """
        Load the vector store from disk.
        Note: Qdrant handles loading automatically when using persistent storage.
        This method is kept for compatibility but doesn't need to do anything.
        
        Args:
            filepath: Path to load the vector store from (not used for Qdrant)
        """
        try:
            logger.info("Qdrant vector store is automatically loaded from persistent storage")
            # Qdrant handles loading automatically when using persistent storage
            # No additional load operation needed
        except Exception as e:
            logger.error(f"Error in load operation: {e}")
            raise
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            logger.info(f"Clearing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "num_documents": collection_info.points_count,
                "model_name": self.model_name,
                "dimension": self.dimension,
                "collection_name": self.collection_name,
                "storage_type": "persistent" if self.use_persistent_storage else "remote",
                "storage_path": self.storage_path if self.use_persistent_storage else None
            }
        except Exception as e:
            logger.error(f"Error getting vector store info: {e}")
            return {
                "num_documents": 0,
                "model_name": self.model_name,
                "dimension": self.dimension,
                "collection_name": self.collection_name,
                "storage_type": "persistent" if self.use_persistent_storage else "remote",
                "storage_path": self.storage_path if self.use_persistent_storage else None,
                "error": str(e)
            }
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete specific documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            logger.info(f"Deleting {len(ids)} documents from vector store")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            logger.info("Documents deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def search_with_filter(self, query: str, filter_condition: Dict[str, Any], n_results: int = 3) -> Dict[str, Any]:
        """
        Search for similar documents with a filter condition.
        
        Args:
            query: Query text
            filter_condition: Filter condition for the search
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query])
            
            # Create filter
            filter_obj = Filter(
                must=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ) for key, value in filter_condition.items()
                ]
            )
            
            # Search with filter
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding[0].tolist(),
                limit=n_results,
                with_payload=True,
                query_filter=filter_obj
            )
            
            # Prepare results
            documents = []
            metadatas = []
            ids = []
            distances = []
            
            for result in search_results:
                documents.append(result.payload.get("text", ""))
                metadata = json.loads(result.payload.get("metadata", "{}"))
                metadatas.append(metadata)
                ids.append(result.id)
                distances.append(1 - result.score)
            
            return {
                "documents": documents,
                "metadatas": metadatas,
                "ids": ids,
                "distances": distances
            }
            
        except Exception as e:
            logger.error(f"Error during filtered search: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "ids": [],
                "distances": []
            } 