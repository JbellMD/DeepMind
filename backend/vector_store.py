#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List, Dict, Any, Optional, Union
import uuid
import json
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

# Define document schema
class Document(BaseModel):
    """Schema for documents stored in the vector database."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class VectorStore:
    """Simple file-based vector store for document embeddings."""
    
    def __init__(self, data_dir: str = "data/vector_store"):
        """Initialize the vector store.
        
        Args:
            data_dir: Directory to store vector data
        """
        self.data_dir = Path(data_dir)
        self.documents_dir = self.data_dir / "documents"
        self.index_path = self.data_dir / "index.json"
        self.metadata_path = self.data_dir / "metadata.json"
        
        # Create directories if they don't exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load index
        self.index = self._load_index()
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized vector store at {self.data_dir}")
        logger.info(f"Found {len(self.index)} documents in the store")
    
    def _load_index(self) -> Dict[str, str]:
        """Load document index from disk."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save document index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f)
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata from disk."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save document metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def add_document(self, document: Document) -> str:
        """Add a document to the vector store.
        
        Args:
            document: Document to add
            
        Returns:
            str: Document ID
        """
        # Save document to disk
        doc_path = self.documents_dir / f"{document.id}.json"
        with open(doc_path, 'w') as f:
            f.write(document.json())
        
        # Update index
        self.index[document.id] = str(doc_path)
        self._save_index()
        
        # Update metadata
        self.metadata[document.id] = {
            "content_preview": document.content[:100] + "..." if len(document.content) > 100 else document.content,
            **document.metadata
        }
        self._save_metadata()
        
        logger.info(f"Added document {document.id} to vector store")
        return document.id
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        if doc_id not in self.index:
            return None
        
        doc_path = self.index[doc_id]
        with open(doc_path, 'r') as f:
            doc_data = json.load(f)
            return Document(**doc_data)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            bool: True if document was deleted, False otherwise
        """
        if doc_id not in self.index:
            return False
        
        # Remove document file
        doc_path = Path(self.index[doc_id])
        if doc_path.exists():
            doc_path.unlink()
        
        # Update index
        del self.index[doc_id]
        self._save_index()
        
        # Update metadata
        if doc_id in self.metadata:
            del self.metadata[doc_id]
            self._save_metadata()
        
        logger.info(f"Deleted document {doc_id} from vector store")
        return True
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        query_embedding = np.array(query_embedding)
        results = []
        
        for doc_id, doc_path in self.index.items():
            try:
                doc = self.get_document(doc_id)
                if doc and doc.embedding:
                    # Calculate cosine similarity
                    doc_embedding = np.array(doc.embedding)
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    
                    results.append({
                        "id": doc_id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity": float(similarity)
                    })
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {str(e)}")
        
        # Sort by similarity (highest first) and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List documents in the vector store.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document metadata
        """
        doc_ids = list(self.index.keys())[offset:offset+limit]
        return [
            {"id": doc_id, **self.metadata.get(doc_id, {})}
            for doc_id in doc_ids
        ]
