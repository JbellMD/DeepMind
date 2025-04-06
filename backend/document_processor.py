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
        separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
    ):
        """Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators to use for splitting, in order of preference
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Handle empty or whitespace-only text
        if not text or text.isspace():
            return []
        
        # If text is shorter than chunk_size, return it as is
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        
        # Try each separator in order
        for separator in self.separators:
            if separator == "":
                # If we've reached the empty separator, split by character
                current_chunk = ""
                for char in text:
                    if len(current_chunk) >= self.chunk_size:
                        chunks.append(current_chunk)
                        current_chunk = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_chunk += char
                
                if current_chunk:
                    chunks.append(current_chunk)
                break
            
            # Split by the current separator
            splits = text.split(separator)
            
            # If we get useful splits, process them
            if len(splits) > 1:
                current_chunk = ""
                
                for split in splits:
                    if not split:
                        continue
                        
                    # If adding this split would exceed chunk_size, add the current chunk to results
                    if len(current_chunk) + len(split) + len(separator) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk)
                        # Start new chunk with overlap
                        current_chunk = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    
                    # Add the split to the current chunk
                    if current_chunk:
                        current_chunk += separator + split
                    else:
                        current_chunk = split
                
                # Add the final chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If we successfully split the text, return the chunks
                if chunks:
                    return chunks
        
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
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process text into documents.
        
        Args:
            text: Text to process
            metadata: Optional metadata to attach to the documents
            
        Returns:
            List of processed documents
        """
        metadata = metadata or {}
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Extract metadata if available
            chunk_metadata = {**metadata}
            for extractor in self.metadata_extractors:
                try:
                    extracted = extractor(chunk)
                    if extracted:
                        chunk_metadata.update(extracted)
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {str(e)}")
            
            # Add chunk index to metadata
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_count"] = len(chunks)
            
            # Create document
            doc = Document(
                content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def process_file(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
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
        
        # Basic metadata from file
        file_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size,
            "created_at": file_path.stat().st_ctime,
            "modified_at": file_path.stat().st_mtime,
        }
        
        # Merge with provided metadata
        if metadata:
            file_metadata.update(metadata)
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.process_text(text, file_metadata)
        except UnicodeDecodeError:
            logger.warning(f"Could not decode file as UTF-8: {file_path}")
            # Try with a different encoding or handle binary files
            return []
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []

# Example metadata extractors
def extract_title_from_text(text: str) -> Dict[str, Any]:
    """Extract a title from the first line of text."""
    lines = text.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        # If first line looks like a title (not too long, no punctuation at end)
        if len(first_line) <= 100 and not first_line.endswith(('.', '?', '!')):
            return {"title": first_line}
    return {}

def extract_code_language(text: str) -> Dict[str, Any]:
    """Attempt to detect programming language from code snippets."""
    # Simple heuristics for language detection
    patterns = {
        "python": r"import\s+[\w\.]+|def\s+\w+\s*\(|class\s+\w+\s*:",
        "javascript": r"const\s+\w+\s*=|let\s+\w+\s*=|function\s+\w+\s*\(|import\s+.*from\s+['\"]",
        "typescript": r"interface\s+\w+|type\s+\w+\s*=|class\s+\w+\s*implements",
        "html": r"<!DOCTYPE\s+html>|<html>|<head>|<body>",
        "css": r"{\s*[\w-]+\s*:\s*[\w-]+\s*;}|@media|@keyframes",
        "sql": r"SELECT\s+.*FROM|INSERT\s+INTO|CREATE\s+TABLE|UPDATE\s+.*SET",
    }
    
    for lang, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return {"language": lang}
    
    return {}
