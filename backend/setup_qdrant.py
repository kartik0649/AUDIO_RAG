#!/usr/bin/env python3
"""
Setup script for migrating from FAISS to Qdrant vector database.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_dependencies():
    """Install required dependencies for Qdrant."""
    print("Installing Qdrant dependencies...")
    
    try:
        # Install qdrant-client and grpcio
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "qdrant-client==1.7.0", "grpcio==1.59.3"
        ])
        print("✅ Qdrant dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def cleanup_faiss_files():
    """Clean up old FAISS files."""
    print("Cleaning up old FAISS files...")
    
    faiss_files = [
        "faiss_knowledge_base.index",
        "faiss_knowledge_base.metadata",
        "faiss_vector_store.py"
    ]
    
    cleaned_files = []
    for file_path in faiss_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                cleaned_files.append(file_path)
                print(f"  ✅ Removed: {file_path}")
            except Exception as e:
                print(f"  ⚠️  Could not remove {file_path}: {e}")
    
    if cleaned_files:
        print(f"✅ Cleaned up {len(cleaned_files)} FAISS files")
    else:
        print("ℹ️  No FAISS files found to clean up")
    
    return True

def create_qdrant_storage_directory():
    """Create Qdrant storage directory."""
    storage_path = "./qdrant_storage"
    
    try:
        os.makedirs(storage_path, exist_ok=True)
        print(f"✅ Created Qdrant storage directory: {storage_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create storage directory: {e}")
        return False

def test_qdrant_installation():
    """Test if Qdrant is properly installed."""
    print("Testing Qdrant installation...")
    
    try:
        # Try to import qdrant_client
        import qdrant_client
        print("✅ qdrant-client imported successfully")
        
        # Try to create a simple client
        from qdrant_client import QdrantClient
        client = QdrantClient(":memory:")  # Use in-memory for testing
        print("✅ QdrantClient created successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Qdrant import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Qdrant test failed: {e}")
        return False

def update_requirements():
    """Update requirements.txt to remove FAISS and add Qdrant."""
    print("Updating requirements.txt...")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"⚠️  {requirements_file} not found, skipping update")
        return True
    
    try:
        # Read current requirements
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        # Remove FAISS and add Qdrant
        updated_lines = []
        faiss_removed = False
        qdrant_added = False
        grpcio_added = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("faiss-cpu"):
                faiss_removed = True
                print("  ✅ Removed faiss-cpu")
                continue
            elif line.startswith("qdrant-client"):
                qdrant_added = True
                updated_lines.append(line)
            elif line.startswith("grpcio"):
                grpcio_added = True
                updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Add Qdrant dependencies if not already present
        if not qdrant_added:
            updated_lines.append("qdrant-client==1.7.0")
            print("  ✅ Added qdrant-client==1.7.0")
        
        if not grpcio_added:
            updated_lines.append("grpcio==1.59.3")
            print("  ✅ Added grpcio==1.59.3")
        
        # Write updated requirements
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(updated_lines) + '\n')
        
        print("✅ requirements.txt updated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update requirements.txt: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Qdrant Vector Database")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Please run this script from the backend directory")
        sys.exit(1)
    
    success = True
    
    # Step 1: Install dependencies
    if not install_dependencies():
        success = False
    
    # Step 2: Test installation
    if success and not test_qdrant_installation():
        success = False
    
    # Step 3: Update requirements.txt
    if success and not update_requirements():
        success = False
    
    # Step 4: Create storage directory
    if success and not create_qdrant_storage_directory():
        success = False
    
    # Step 5: Clean up old FAISS files
    if success:
        cleanup_faiss_files()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Qdrant setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python test_qdrant.py' to test the implementation")
        print("2. Start your application with 'python main.py'")
        print("3. The vector store will automatically use persistent storage")
        sys.exit(0)
    else:
        print("💥 Qdrant setup failed!")
        print("\nPlease check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 