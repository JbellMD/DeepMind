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
    """A simple file-based vector store for document storage and retrieval."""
    
    def __init__(self, store_dir: str):
        """Initialize the vector store.
        
        Args:
            store_dir: Directory to store vector data
        """
        self.store_dir = Path(store_dir)
        self.docs_dir = self.store_dir / "documents"
        self.index_path = self.store_dir / "index.json"
        
        # Create directories if they don't exist
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the document index from disk."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save the document index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)
    
    def _save_document(self, doc: Document):
        """Save a document and its embedding to disk."""
        doc_path = self.docs_dir / f"{doc.id}.json"
        with open(doc_path, "w") as f:
            json.dump(doc.dict(), f, indent=2)
    
    def add_document(self, doc: Document) -> str:
        """Add a document to the vector store.
        
        Args:
            doc: Document to add
            
        Returns:
            str: Document ID
        """
        if not doc.embedding:
            raise ValueError("Document must have an embedding")
        
        # Save document
        self._save_document(doc)
        
        # Update index
        self.index[doc.id] = {
            "metadata": doc.metadata,
            "embedding": doc.embedding
        }
        self._save_index()
        
        logger.info(f"Added document {doc.id} to vector store")
        return doc.id
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Optional[Document]: Document if found, None otherwise
        """
        doc_path = self.docs_dir / f"{doc_id}.json"
        if not doc_path.exists():
            return None
            
        with open(doc_path, "r") as f:
            return Document(**json.load(f))
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            bool: True if document was deleted, False if not found
        """
        doc_path = self.docs_dir / f"{doc_id}.json"
        if not doc_path.exists():
            return False
            
        # Remove document file
        doc_path.unlink()
        
        # Update index
        if doc_id in self.index:
            del self.index[doc_id]
            self._save_index()
            
        logger.info(f"Deleted document {doc_id} from vector store")
        return True
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of documents with similarity scores
        """
        if not self.index:
            return []
            
        # Convert query to numpy array
        query_vec = np.array(query_embedding)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_data in self.index.items():
            doc_vec = np.array(doc_data["embedding"])
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((doc_id, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Get full documents
        results = []
        for doc_id, score in top_results:
            doc = self.get_document(doc_id)
            if doc:
                results.append({
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
        
        return results
    
    def list_documents(self) -> List[str]:
        """List all document IDs in the store.
        
        Returns:
            List[str]: List of document IDs
        """
        return list(self.index.keys())
