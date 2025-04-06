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
            self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
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
            # Truncate text if it's too long (most models have a token limit)
            if len(text) > 10000:
                logger.warning(f"Text is too long ({len(text)} chars), truncating to 10000 chars")
                text = text[:10000]
            
            # Generate embedding
            embedding = self.model.encode(text)
            
            # Convert to list of floats
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * self.get_embedding_dimension()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Truncate texts if they're too long
            processed_texts = []
            for text in texts:
                if len(text) > 10000:
                    logger.warning(f"Text is too long ({len(text)} chars), truncating to 10000 chars")
                    processed_texts.append(text[:10000])
                else:
                    processed_texts.append(text)
            
            # Generate embeddings in batch
            embeddings = self.model.encode(processed_texts)
            
            # Convert to list of lists
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero vectors as fallback
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in range(len(texts))]
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of documents with embeddings
        """
        # Extract text from documents
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to documents
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
        
        return documents
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query.
        
        This is the same as generate_embedding, but may be overridden
        in subclasses if query embeddings should be handled differently.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        return self.generate_embedding(query)


class HuggingFaceEmbeddingService(EmbeddingService):
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
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get the embeddings
                embeddings = outputs.last_hidden_state
                
                # Apply pooling strategy
                if self.pooling_strategy == "cls":
                    # Use [CLS] token embedding
                    pooled_embedding = embeddings[:, 0, :]
                elif self.pooling_strategy == "mean":
                    # Mean pooling
                    attention_mask = inputs["attention_mask"]
                    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask, 1)
                    sum_mask = torch.sum(mask, 1)
                    pooled_embedding = sum_embeddings / sum_mask
                elif self.pooling_strategy == "max":
                    # Max pooling
                    attention_mask = inputs["attention_mask"]
                    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings[mask == 0] = -1e9  # Set padding tokens to large negative value
                    pooled_embedding = torch.max(embeddings, 1)[0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
                
                # Convert to numpy and then to list
                embedding_np = pooled_embedding.cpu().numpy()[0]
                
                # Normalize the embedding
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    embedding_np = embedding_np / norm
                
                return embedding_np.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * self.get_embedding_dimension()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self.model.config.hidden_size
