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
            document_processor: Document processor for text chunking
            embedding_service: Embedding service for generating embeddings
            max_context_documents: Maximum number of documents to include in context
            similarity_threshold: Minimum similarity score for retrieved documents
        """
        self.vector_store = vector_store or VectorStore()
        self.document_processor = document_processor or DocumentProcessor()
        self.embedding_service = embedding_service or EmbeddingService()
        self.max_context_documents = max_context_documents
        self.similarity_threshold = similarity_threshold
        
        logger.info("Initialized RAG service")
    
    async def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add a document to the RAG system.
        
        Args:
            content: Document content
            metadata: Optional metadata
            
        Returns:
            List of document IDs
        """
        # Process document into chunks
        documents = self.document_processor.process_text(content, metadata)
        
        # Generate embeddings for documents
        documents = self.embedding_service.embed_documents(documents)
        
        # Add documents to vector store
        doc_ids = []
        for doc in documents:
            doc_id = self.vector_store.add_document(doc)
            doc_ids.append(doc_id)
        
        return doc_ids
    
    async def add_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add a file to the RAG system.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata
            
        Returns:
            List of document IDs
        """
        # Process file into documents
        documents = self.document_processor.process_file(file_path, metadata)
        
        # Generate embeddings for documents
        documents = self.embedding_service.embed_documents(documents)
        
        # Add documents to vector store
        doc_ids = []
        for doc in documents:
            doc_id = self.vector_store.add_document(doc)
            doc_ids.append(doc_id)
        
        return doc_ids
    
    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with similarity scores
        """
        top_k = top_k or self.max_context_documents
        
        # Generate query embedding
        query_embedding = self.embedding_service.get_query_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Filter by similarity threshold
        results = [r for r in results if r["similarity"] >= self.similarity_threshold]
        
        return results
    
    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string for the LLM.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        
        for i, doc in enumerate(results):
            # Format source information
            source_info = ""
            if "source" in doc["metadata"]:
                source_info = f"Source: {doc['metadata']['source']}"
            elif "title" in doc["metadata"]:
                source_info = f"Title: {doc['metadata']['title']}"
            
            # Format document content
            content = doc["content"].strip()
            
            # Add to context parts
            context_parts.append(f"[Document {i+1}] {source_info}\n{content}\n")
        
        # Join all parts
        context = "\n".join(context_parts)
        
        return context
    
    async def generate_prompt_with_context(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a prompt with retrieved context for the query.
        
        Args:
            query: User query
            system_prompt: System prompt template
            chat_history: Optional chat history
            
        Returns:
            Tuple of (augmented system prompt, retrieved documents)
        """
        # Retrieve relevant documents
        retrieved_docs = await self.retrieve(query)
        
        # Format context
        context = self.format_retrieved_context(retrieved_docs)
        
        # Create augmented system prompt
        if system_prompt is None:
            system_prompt = """You are DeepMind AI, an advanced AI assistant built on a fine-tuned DeepSeek model.
You are designed to be helpful, harmless, and honest in all your interactions.
You excel at providing detailed, thoughtful responses while maintaining a friendly and conversational tone."""
        
        # Add context to system prompt if we have retrieved documents
        if context:
            augmented_prompt = f"""{system_prompt}

I'll provide you with some relevant information to help answer the user's question.
Use this information to inform your response, but you don't need to use all of it.
If the information doesn't seem relevant, rely on your own knowledge.

Here's the relevant information:
{context}

Remember to answer the user's question directly and concisely."""
        else:
            augmented_prompt = system_prompt
        
        return augmented_prompt, retrieved_docs
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the RAG system.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted, False otherwise
        """
        return self.vector_store.delete_document(doc_id)
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List documents in the RAG system.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document metadata
        """
        return self.vector_store.list_documents(limit=limit, offset=offset)
