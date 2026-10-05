from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings


def get_embeddings_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def chunk_documents_recursive(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def chunk_documents_semantic(documents: List[Document], breakpoint_threshold_type: str = "percentile") -> List[Document]:
    embeddings = get_embeddings_model()
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
    )
    return splitter.split_documents(documents)


def chunk_documents(documents: List[Document], method: str) -> List[Document]:
    if method == "semantic":
        return chunk_documents_semantic(documents)
    return chunk_documents_recursive(documents)