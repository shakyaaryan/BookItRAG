import os
import uuid
import tempfile
from typing import List, BinaryIO
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def save_uploaded_files(files: List[BinaryIO], filenames: List[str]) -> List[tuple[str, str]]:
    saved_files = []
    for file_content, filename in zip(files, filenames):
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content.read())
            saved_files.append((tmp_file.name, filename))
    return saved_files


def load_documents(file_paths: List[tuple[str, str]]) -> List[Document]:
    documents = []
    for file_path, original_name in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            continue
        
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = original_name
        documents.extend(docs)
    return documents


def cleanup_temp_files(file_paths: List[tuple[str, str]]) -> None:
    for file_path, _ in file_paths:
        try:
            os.unlink(file_path)
        except OSError:
            pass


def generate_doc_id() -> str:
    return f"doc_{uuid.uuid4().hex[:8]}"