#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from vector_store import Document

class TextSplitter:
    """Split text into chunks for embedding and retrieval."""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: String to use for splitting text into chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # First split by separator
        splits = text.split(self.separator)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
                
            split_size = len(split)
            
            if current_size + split_size <= self.chunk_size:
                current_chunk.append(split)
                current_size += split_size
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))
                
                # Start new chunk with overlap
                if current_chunk and self.chunk_overlap > 0:
                    # Calculate how many previous splits to keep for overlap
                    overlap_size = 0
                    overlap_chunks = []
                    for chunk in reversed(current_chunk):
                        if overlap_size + len(chunk) <= self.chunk_overlap:
                            overlap_chunks.insert(0, chunk)
                            overlap_size += len(chunk)
                        else:
                            break
                    current_chunk = overlap_chunks
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
                
                # Add current split to new chunk
                current_chunk.append(split)
                current_size += split_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks

class DocumentProcessor:
    """Process documents for the vector store."""
    
    def __init__(
        self,
        text_splitter: Optional[TextSplitter] = None,
        metadata_extractors: Optional[List[callable]] = None
    ):
        """Initialize the document processor.
        
        Args:
            text_splitter: Text splitter to use
            metadata_extractors: List of functions to extract metadata from documents
        """
        self.text_splitter = text_splitter or TextSplitter()
        self.metadata_extractors = metadata_extractors or []
    
    def process_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process text into documents.
        
        Args:
            text: Text to process
            metadata: Optional metadata to attach to the documents
            
        Returns:
            List of processed documents
        """
        # Initialize metadata
        metadata = metadata or {}
        
        # Extract additional metadata
        for extractor in self.metadata_extractors:
            try:
                extracted = extractor(text)
                if extracted:
                    metadata.update(extracted)
            except Exception as e:
                logger.warning(f"Metadata extractor failed: {str(e)}")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            # Add chunk-specific metadata
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                **metadata
            }
            
            # Create document
            doc = Document(
                content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def process_file(
        self, 
        file_path: Union[str, Path], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process a file into documents.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata to attach to the documents
            
        Returns:
            List of processed documents
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Initialize metadata with file info
        metadata = metadata or {}
        file_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix.lower()[1:] if file_path.suffix else None,
            "created": os.path.getctime(file_path),
            "modified": os.path.getmtime(file_path),
            **metadata
        }
        
        # Read and process file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            return self.process_text(text, metadata=file_metadata)

# Example metadata extractors
def extract_title_from_text(text: str) -> Optional[Dict[str, str]]:
    """Extract a title from the first line of text."""
    if not text:
        return None
    
    # Get first non-empty line
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line:
            return {"title": line}
    return None

def extract_code_language(text: str) -> Optional[Dict[str, str]]:
    """Attempt to detect programming language from code snippets."""
    # Simple language detection based on file patterns
    patterns = {
        "python": r"(import\s+[\w.]+|from\s+[\w.]+\s+import|def\s+\w+\s*\(|class\s+\w+:)",
        "javascript": r"(const\s+\w+\s*=|let\s+\w+\s*=|function\s+\w+\s*\(|class\s+\w+\s*{)",
        "typescript": r"(interface\s+\w+|type\s+\w+\s*=|export\s+class)",
        "html": r"(<html|<!DOCTYPE\s+html|<head|<body)",
        "css": r"(@media|@import|[.#]\w+\s*{)",
        "sql": r"(SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE|ALTER\s+TABLE)\s"
    }
    
    for lang, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return {"language": lang}
    
    return None
