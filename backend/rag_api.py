#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from typing import List, Dict, Any, Optional, Union

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from rag_service import RAGService
from vector_store import Document
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from model import generate_response, stream_response

# Initialize RAG components
vector_store_dir = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Create router
router = APIRouter(prefix="/rag", tags=["RAG"])

# Initialize services
document_processor = DocumentProcessor()
embedding_service = EmbeddingService(model_name=embedding_model)
rag_service = RAGService(
    data_dir=vector_store_dir,
    document_processor=document_processor,
    embedding_service=embedding_service
)

# Pydantic models
class DocumentUpload(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DocumentResponse(BaseModel):
    id: str
    content_preview: str
    metadata: Dict[str, Any]

class RAGChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    use_rag: Optional[bool] = True

class RAGChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    created: int = Field(default_factory=lambda: int(time.time()))

# API endpoints
@router.post("/documents", response_model=DocumentResponse)
async def upload_document(document: DocumentUpload):
    """Upload a document to the RAG system."""
    try:
        doc_id = await rag_service.add_document(document.content, document.metadata)
        doc = rag_service.vector_store.get_document(doc_id)
        
        if not doc:
            raise HTTPException(status_code=500, detail="Failed to retrieve document after upload")
        
        return DocumentResponse(
            id=doc_id,
            content_preview=doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
            metadata=doc.metadata
        )
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@router.post("/documents/file", response_model=List[DocumentResponse])
async def upload_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """Upload a file to the RAG system."""
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode("utf-8")
        
        # Parse metadata if provided
        meta_dict = {}
        if metadata:
            import json
            meta_dict = json.loads(metadata)
        
        # Add file metadata
        meta_dict["filename"] = file.filename
        meta_dict["content_type"] = file.content_type
        
        # Process document
        doc_ids = await rag_service.add_document(content_str, meta_dict)
        
        # Get documents
        responses = []
        for doc_id in doc_ids:
            doc = rag_service.vector_store.get_document(doc_id)
            if doc:
                responses.append(DocumentResponse(
                    id=doc_id,
                    content_preview=doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                    metadata=doc.metadata
                ))
        
        return responses
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List documents in the RAG system."""
    try:
        docs = rag_service.vector_store.list_documents(limit=limit, offset=offset)
        return [
            DocumentResponse(
                id=doc["id"],
                content_preview=doc.get("content_preview", ""),
                metadata={k: v for k, v in doc.items() if k not in ["id", "content_preview"]}
            )
            for doc in docs
        ]
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get a document from the RAG system."""
    try:
        doc = rag_service.vector_store.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        return DocumentResponse(
            id=doc_id,
            content_preview=doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
            metadata=doc.metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the RAG system."""
    try:
        success = rag_service.vector_store.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        return {"status": "success", "message": f"Document {doc_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.post("/chat", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    """Generate a response using RAG."""
    try:
        # Convert chat history to the expected format
        messages = request.chat_history or []
        
        # Add the current query
        messages.append({"role": "user", "content": request.query})
        
        if request.use_rag:
            # Generate augmented prompt with retrieved context
            system_prompt, retrieved_docs = await rag_service.generate_prompt_with_context(
                query=request.query,
                system_prompt=request.system_prompt,
                chat_history=request.chat_history
            )
            
            # Generate response
            response = await generate_response(
                messages=messages,
                system_prompt=system_prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            # Format sources
            sources = [
                {
                    "id": doc["id"],
                    "content_preview": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": doc["similarity"]
                }
                for doc in retrieved_docs
            ]
        else:
            # Generate response without RAG
            response = await generate_response(
                messages=messages,
                system_prompt=request.system_prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )
            sources = []
        
        return RAGChatResponse(
            response=response,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error generating RAG response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating RAG response: {str(e)}")

@router.post("/chat/stream")
async def rag_chat_stream(request: RAGChatRequest):
    """Stream a response using RAG."""
    try:
        # Convert chat history to the expected format
        messages = request.chat_history or []
        
        # Add the current query
        messages.append({"role": "user", "content": request.query})
        
        if request.use_rag:
            # Generate augmented prompt with retrieved context
            system_prompt, retrieved_docs = await rag_service.generate_prompt_with_context(
                query=request.query,
                system_prompt=request.system_prompt,
                chat_history=request.chat_history
            )
            
            # Stream response
            async def stream_generator():
                async for chunk in stream_response(
                    messages=messages,
                    system_prompt=system_prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature
                ):
                    yield f"data: {chunk}\n\n"
                
                # Send sources at the end
                import json
                sources = [
                    {
                        "id": doc["id"],
                        "content_preview": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"],
                        "metadata": doc["metadata"],
                        "similarity": doc["similarity"]
                    }
                    for doc in retrieved_docs
                ]
                yield f"data: [SOURCES]{json.dumps(sources)}[/SOURCES]\n\n"
                yield "data: [DONE]\n\n"
        else:
            # Stream response without RAG
            async def stream_generator():
                async for chunk in stream_response(
                    messages=messages,
                    system_prompt=request.system_prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature
                ):
                    yield f"data: {chunk}\n\n"
                
                yield f"data: [SOURCES][]\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error streaming RAG response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error streaming RAG response: {str(e)}")
