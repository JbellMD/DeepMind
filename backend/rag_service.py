#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger

from vector_store import VectorStore, Document
from document_processor import DocumentProcessor, TextSplitter
from embedding_service import EmbeddingService

class RAGService:
    """Retrieval-Augmented Generation service."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        document_processor: Optional[DocumentProcessor] = None,
        embedding_service: Optional[EmbeddingService] = None,
        max_context_documents: int = 5,
        similarity_threshold: float = 0.6
    ):
        """Initialize the RAG service.
        
        Args:
            vector_store: Vector store for document storage and retrieval
            document_processor: Document processor for text splitting
            embedding_service: Service for generating embeddings
            max_context_documents: Maximum number of documents to include in context
            similarity_threshold: Minimum similarity score for retrieved documents
        """
        # Initialize components with defaults if not provided
        self.vector_store = vector_store or VectorStore(os.path.join("data", "vector_store"))
        self.document_processor = document_processor or DocumentProcessor()
        self.embedding_service = embedding_service or EmbeddingService()
        
        self.max_context_documents = max_context_documents
        self.similarity_threshold = similarity_threshold
        
        logger.info("Initialized RAG service")
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document to the RAG system.
        
        Args:
            content: Document content
            metadata: Optional metadata for the document
            
        Returns:
            Document ID
        """
        try:
            # Process document
            documents = self.document_processor.process_text(content, metadata)
            
            # Generate embeddings
            documents = self.embedding_service.embed_documents(documents)
            
            # Store documents
            doc_ids = []
            for doc in documents:
                doc_id = self.vector_store.add_document(doc)
                doc_ids.append(doc_id)
            
            logger.info(f"Added document with {len(documents)} chunks")
            return doc_ids[0]  # Return ID of first chunk
            
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise RuntimeError(f"Failed to add document: {str(e)}")
    
    def add_file(self, file_path: Union[str, os.PathLike], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a file to the RAG system.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata for the document
            
        Returns:
            Document ID
        """
        try:
            # Process file
            documents = self.document_processor.process_file(file_path, metadata)
            
            # Generate embeddings
            documents = self.embedding_service.embed_documents(documents)
            
            # Store documents
            doc_ids = []
            for doc in documents:
                doc_id = self.vector_store.add_document(doc)
                doc_ids.append(doc_id)
            
            logger.info(f"Added file {file_path} with {len(documents)} chunks")
            return doc_ids[0]  # Return ID of first chunk
            
        except Exception as e:
            logger.error(f"Failed to add file: {str(e)}")
            raise RuntimeError(f"Failed to add file: {str(e)}")
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the RAG system.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted
        """
        try:
            return self.vector_store.delete_document(doc_id)
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            raise RuntimeError(f"Failed to delete document: {str(e)}")
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Maximum number of documents to retrieve (default: max_context_documents)
            min_similarity: Minimum similarity score (default: similarity_threshold)
            
        Returns:
            List of relevant documents with similarity scores
        """
        try:
            # Use default values if not specified
            top_k = top_k or self.max_context_documents
            min_similarity = min_similarity or self.similarity_threshold
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Retrieve similar documents
            results = self.vector_store.search(query_embedding, top_k=top_k)
            
            # Filter by similarity threshold
            results = [
                result for result in results 
                if result["score"] >= min_similarity
            ]
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            raise RuntimeError(f"Failed to retrieve documents: {str(e)}")
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        try:
            return self.vector_store.get_document(doc_id)
        except Exception as e:
            logger.error(f"Failed to get document: {str(e)}")
            raise RuntimeError(f"Failed to get document: {str(e)}")
    
    def list_documents(self) -> List[str]:
        """List all document IDs in the system.
        
        Returns:
            List of document IDs
        """
        try:
            return self.vector_store.list_documents()
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise RuntimeError(f"Failed to list documents: {str(e)}")
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents with scores
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        # Sort by similarity score
        documents.sort(key=lambda x: x["score"], reverse=True)
        
        # Format context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc["content"].strip()
            score = doc["score"]
            metadata = doc.get("metadata", {})
            
            # Add source info if available
            source_info = ""
            if "source" in metadata:
                source_info = f" (Source: {metadata['source']})"
            
            context_parts.append(
                f"[Document {i}{source_info}, Relevance: {score:.2f}]\n{content}\n"
            )
        
        return "\n".join(context_parts)
