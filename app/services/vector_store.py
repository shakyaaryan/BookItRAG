from typing import List, Dict, Any, Optional
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings


class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY.get_secret_value())
        self.index_name = settings.PINECONE_INDEX_NAME
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self) -> None:
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def upsert_documents(
        self,
        documents: List[Document],
        doc_id: str,
        ingested_at: Optional[datetime] = None,
    ) -> int:
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)

        ingested_at_str = ingested_at.isoformat() if ingested_at else datetime.utcnow().isoformat()

        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            metadata = {
                "doc_id": doc_id,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "text": doc.page_content,
                "ingested_at": ingested_at_str,
            }
            vectors.append({
                "id": f"{doc_id}_{i}",
                "values": embedding,
                "metadata": metadata,
            })

        if vectors:
            self.index.upsert(vectors=vectors)
        return len(vectors)

    def similarity_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embeddings.embed_query(query)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )
        return results.matches if results.matches else []


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store