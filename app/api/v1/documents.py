from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from app.database import get_db
from app.models import DocumentMetadata
from app.schemas import DocumentProcessed


router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentListItem(DocumentProcessed):
    """Document list item — inherits from DocumentProcessed."""
    pass


@router.get(
    "",
    response_model=List[DocumentListItem],
    summary="List ingested documents",
    description="""Retrieve a paginated list of all ingested documents, ordered by ingestion date (newest first).

**Pagination:**
- `limit`: Number of items per page (1-100, default 50)
- `offset`: Number of items to skip (default 0)

**Response:** Array of documents with:
- `doc_id`: Unique document identifier
- `file_name`: Original filename
- `chunking_method`: "recursive" or "semantic"
- `ingested_at`: ISO 8601 timestamp of ingestion
""",
)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=100, description="Items per page (1-100)"),
    offset: int = Query(default=0, ge=0, description="Items to skip"),
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