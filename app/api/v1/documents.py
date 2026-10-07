from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from app.database import get_db
from app.models import DocumentMetadata
from app.schemas import DocumentProcessed


router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentListItem(DocumentProcessed):
    """Extended document info for listing."""
    pass


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
        result.append(DocumentListItem(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            chunking_method=doc.chunking_method.value if doc.chunking_method else "unknown",
            ingested_at=doc.ingested_at,
        ))
    
    return result