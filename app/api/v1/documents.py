from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.database import get_db
from app.models import DocumentMetadata
from app.schemas import DocumentProcessed


router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentListItem(DocumentProcessed):
    """Extended document info for listing."""
    chunk_count: int = 0


@router.get("", response_model=List[DocumentListItem])
async def list_documents(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """List all ingested documents with pagination."""
    documents = db.query(DocumentMetadata).order_by(
        DocumentMetadata.ingested_at.desc()
    ).offset(offset).limit(limit).all()
    
    result = []
    for doc in documents:
        # Count chunks for this document from vector store metadata
        # For now, we don't have direct chunk count in SQL, return what we have
        result.append(DocumentListItem(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            chunking_method=doc.chunking_method.value if doc.chunking_method else "unknown",
            chunks_created=0,  # Would need vector store query for accurate count
            ingested_at=doc.ingested_at,
            chunk_count=0,
        ))
    
    return result


@router.get("/{doc_id}", response_model=DocumentListItem)
async def get_document(
    doc_id: str,
    db: Session = Depends(get_db),
):
    """Get a specific document by ID."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    
    if not doc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentListItem(
        doc_id=doc.doc_id,
        file_name=doc.file_name,
        chunking_method=doc.chunking_method.value if doc.chunking_method else "unknown",
        chunks_created=0,
        ingested_at=doc.ingested_at,
        chunk_count=0,
    )