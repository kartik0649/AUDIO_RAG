# Migration from FAISS to Qdrant Vector Database

This document describes the migration from FAISS to Qdrant vector database for the Audio RAG system.

## Overview

The system has been migrated from FAISS to Qdrant to provide:
- **Persistent storage**: Data is automatically saved and persists between application restarts
- **Better scalability**: Qdrant is designed for production use with better performance characteristics
- **Advanced filtering**: Support for metadata-based filtering during search
- **REST API**: Built-in HTTP API for external access if needed

## Key Changes

### 1. Dependencies
- **Removed**: `faiss-cpu==1.7.4`
- **Added**: `qdrant-client==1.7.0`, `grpcio==1.59.3`

### 2. Vector Store Implementation
- **Old**: `faiss_vector_store.py` (FAISSVectorStore class)
- **New**: `qdrant_vector_store.py` (QdrantVectorStore class)

### 3. Storage
- **Old**: Manual file-based storage (`faiss_knowledge_base.index`, `faiss_knowledge_base.metadata`)
- **New**: Automatic persistent storage in `qdrant_storage/` directory

### 4. Configuration
- **Old**: `VECTOR_STORE_PATH = "faiss_knowledge_base"`
- **New**: `VECTOR_STORE_PATH = "qdrant_storage"`

## Migration Steps

### Step 1: Run the Setup Script
```bash
cd backend
python setup_qdrant.py
```

This script will:
- Install Qdrant dependencies
- Update requirements.txt
- Create storage directory
- Clean up old FAISS files
- Test the installation

### Step 2: Test the Implementation
```bash
python test_qdrant.py
```

This will verify that:
- Qdrant client can be initialized
- Documents can be added and retrieved
- Search functionality works correctly
- Persistent storage is working

### Step 3: Start the Application
```bash
python main.py
```

The application will now use Qdrant with persistent storage.

## New Features

### 1. Automatic Persistence
Data is automatically saved to disk and persists between application restarts. No manual save/load operations needed.

### 2. Advanced Filtering
```python
# Search with metadata filters
filter_condition = {"source": "specific_file"}
results = vector_store.search_with_filter(query, filter_condition, n_results=3)
```

### 3. Better Metadata Handling
Metadata is stored as JSON in the payload, allowing for more flexible querying and filtering.

### 4. Collection Management
```python
# Get collection information
info = vector_store.get_info()

# Delete specific documents
vector_store.delete_documents(["doc_id_1", "doc_id_2"])

# Clear entire collection
vector_store.clear()
```

## Configuration Options

The QdrantVectorStore constructor supports several configuration options:

```python
vector_store = QdrantVectorStore(
    model_name="all-MiniLM-L6-v2",      # Embedding model
    dimension=384,                       # Vector dimension
    collection_name="audio_rag_documents", # Collection name
    use_persistent_storage=True,         # Use local storage
    storage_path="./qdrant_storage"      # Storage directory
)
```

## Performance Considerations

### Memory Usage
- Qdrant uses more memory than FAISS for small datasets
- For large datasets, Qdrant provides better memory management
- Consider using remote Qdrant server for very large datasets

### Storage
- Qdrant storage is larger than FAISS due to additional metadata and indexing
- Storage grows linearly with document count
- Consider regular backups of the `qdrant_storage` directory

### Search Performance
- Qdrant provides better search performance for large datasets
- Supports approximate nearest neighbor search with configurable accuracy
- Built-in support for filtering during search

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'qdrant_client'
   ```
   Solution: Run `pip install qdrant-client==1.7.0 grpcio==1.59.3`

2. **Storage Permission Errors**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   Solution: Ensure write permissions to the storage directory

3. **Port Conflicts**
   ```
   Connection refused
   ```
   Solution: Qdrant uses port 6333 by default. Ensure it's available.

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger("qdrant_client").setLevel(logging.DEBUG)
```

### Backup and Restore

To backup your vector store:
```bash
# Copy the entire storage directory
cp -r qdrant_storage qdrant_storage_backup
```

To restore:
```bash
# Stop the application
# Replace the storage directory
rm -rf qdrant_storage
cp -r qdrant_storage_backup qdrant_storage
# Restart the application
```

## Rollback Plan

If you need to rollback to FAISS:

1. Restore the old `faiss_vector_store.py` file
2. Update `main.py` to import `FAISSVectorStore` instead of `QdrantVectorStore`
3. Reinstall FAISS: `pip install faiss-cpu==1.7.4`
4. Remove Qdrant dependencies: `pip uninstall qdrant-client grpcio`

## Support

For issues with the migration:
1. Check the troubleshooting section above
2. Review the test output from `test_qdrant.py`
3. Check the application logs for detailed error messages
4. Ensure all dependencies are properly installed 