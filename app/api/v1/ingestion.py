from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import tempfile
import os

from app.database import get_db
from app.models import DocumentMetadata, ChunkingMethod
from app.schemas import DocumentIngestResponse, DocumentProcessed
from app.services.document_service import (
    load_documents,
    cleanup_temp_files,
    generate_doc_id,
)
from app.services.chunking_service import chunk_documents
from app.services.vector_store import get_vector_store


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    summary="Ingest documents for RAG",
    description="""Upload PDF or TXT files to be processed, chunked, embedded, and stored in Pinecone for RAG retrieval.

**Process:**
1. Files are saved to temporary storage
2. Documents are loaded and split into chunks (recursive or semantic)
3. Embeddings are generated via HuggingFace sentence-transformers
4. Vectors are upserted to Pinecone with document metadata
5. Document metadata is persisted to SQLite

**Supported formats:** PDF (.pdf), Text (.txt)

**Chunking methods:**
- `recursive`: RecursiveCharacterTextSplitter (default, good for general docs)
- `semantic`: SemanticChunker (uses embeddings for semantic boundaries)

**Returns:** List of processed files with doc_id, file_name, chunking_method, and ingestion timestamp.
""",
)
async def ingest_documents(
    chunking_method: str = Form(
        default="recursive",
        description="Chunking strategy: 'recursive' or 'semantic'",
    ),
    files: List[UploadFile] = File(
        ..., description="One or more PDF/TXT files to ingest"
    ),
    db: Session = Depends(get_db),
):
    if chunking_method not in ["recursive", "semantic"]:
        raise HTTPException(status_code=400, detail="Invalid chunking method")

    file_contents = []
    filenames = []
    for file in files:
        if not file.filename.endswith((".pdf", ".txt")):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
        content = await file.read()
        file_contents.append(content)
        filenames.append(file.filename)

    saved_files = []
    for content, filename in zip(file_contents, filenames):
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            saved_files.append((tmp_file.name, filename))

    try:
        documents = load_documents(saved_files)
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents found")

        chunked_docs = chunk_documents(documents, chunking_method)

        vector_store = get_vector_store()
        processed_files = []

        for file_path, filename in saved_files:
            doc_id = generate_doc_id()
            file_docs = [d for d in chunked_docs if d.metadata.get("source") == filename]

            if file_docs:
                chunks_created = vector_store.upsert_documents(file_docs, doc_id)

                metadata = DocumentMetadata(
                    doc_id=doc_id,
                    file_name=filename,
                    chunking_method=ChunkingMethod(chunking_method),
                )
                db.add(metadata)
                db.commit()
                db.refresh(metadata)

                processed_files.append(DocumentProcessed(
                    doc_id=doc_id,
                    file_name=filename,
                    chunking_method=chunking_method,
                    ingested_at=metadata.ingested_at,
                ))

        return DocumentIngestResponse(
            status="success",
            processed_files=processed_files,
        )

    finally:
        cleanup_temp_files(saved_files)