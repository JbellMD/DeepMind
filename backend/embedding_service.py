#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from loguru import logger

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from vector_store import Document

class EmbeddingService:
    """Service for generating embeddings from text."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to run the model on (cuda, cpu, mps)
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        logger.info(f"Using device {self.device} for embeddings")
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir, device=self.device)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.encode(text, convert_to_tensor=True)
                return embedding.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Generate embeddings in batch
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_tensor=True)
                return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of documents with embeddings
        """
        try:
            # Extract text from documents
            texts = [doc.content for doc in documents]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            return documents
        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise RuntimeError(f"Failed to embed documents: {str(e)}")

class HuggingFaceEmbeddingService:
    """Embedding service using HuggingFace models directly."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pooling_strategy: str = "mean"
    ):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to run the model on (cuda, cpu, mps)
            cache_dir: Directory to cache models
            pooling_strategy: Strategy for pooling token embeddings (mean, cls, max)
        """
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        logger.info(f"Using device {self.device} for embeddings")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            self.model.to(self.device)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text using HuggingFace model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get embeddings based on pooling strategy
                if self.pooling_strategy == "cls":
                    embeddings = outputs.last_hidden_state[:, 0]
                elif self.pooling_strategy == "mean":
                    attention_mask = inputs["attention_mask"]
                    embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
                    embeddings = embeddings / attention_mask.sum(-1, keepdim=True)
                elif self.pooling_strategy == "max":
                    attention_mask = inputs["attention_mask"]
                    embeddings = outputs.last_hidden_state.masked_fill(~attention_mask.unsqueeze(-1).bool(), -1e9)
                    embeddings = embeddings.max(dim=1)[0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
                
                return embeddings[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self.model.config.hidden_size
