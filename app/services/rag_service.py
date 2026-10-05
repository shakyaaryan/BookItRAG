from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import settings
from app.services.vector_store import get_vector_store
from app.services.memory_service import get_memory_service


class RAGService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            temperature=0.3,
        )
        self.vector_store = get_vector_store()
        self.memory_service = get_memory_service()

    def _extract_content(self, response) -> str:
        """Extract text content from Gemini response (handles both string and list formats)."""
        content = response.content
        if isinstance(content, list):
            return "".join(block.text for block in content if hasattr(block, 'text'))
        return content.strip() if content else ""

    def _format_history(self, history: List[Dict[str, Any]]) -> List:
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def _rephrase_query(self, query: str, history: List[Dict[str, Any]]) -> str:
        if not history:
            return query

        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in history[-6:]
        ])

        prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures all relevant context.

Conversation History:
{history_text}

Follow-up Question: {query}

Standalone Question:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._extract_content(response)

    def _retrieve_context(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.vector_store.similarity_search(query, top_k)

    def _prioritize_latest(self, docs: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """Sort documents by ingested_at descending (latest first) and return top_k."""
        def get_ingested_at(doc):
            meta = doc.get("metadata", {})
            ingested_str = meta.get("ingested_at")
            if ingested_str:
                try:
                    return datetime.fromisoformat(ingested_str)
                except ValueError:
                    pass
            return datetime.min
        
        sorted_docs = sorted(docs, key=get_ingested_at, reverse=True)
        return sorted_docs[:top_k]

    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "No relevant documents found."

        context_parts = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            source = metadata.get("source", "unknown")
            page = metadata.get("page", 0)
            text = metadata.get("text", "")
            ingested_at = metadata.get("ingested_at", "")
            context_parts.append(f"[Source: {source}, Page: {page}, Ingested: {ingested_at}]\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        session_id: str = "",
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        history = self.memory_service.get_history(session_id)
        standalone_query = self._rephrase_query(query, history)
        retrieved_docs = self._retrieve_context(standalone_query)
        prioritized_docs = self._prioritize_latest(retrieved_docs)
        context = self._format_context(prioritized_docs)

        history_messages = self._format_history(history)

        system_prompt = """You are a helpful AI assistant that answers questions about job postings based on the provided context.
Use only the information from the context to answer. If the context doesn't contain enough information, say so.
Be concise and helpful. Cite sources using [Source: filename, Page: X] format when referencing information.
Prioritize information from the most recently ingested documents."""

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(history_messages)
        messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {standalone_query}"))

        response = self.llm.invoke(messages)
        answer = self._extract_content(response)

        sources = []
        for doc in prioritized_docs:
            metadata = doc.metadata
            sources.append({
                "file_name": metadata.get("source", "unknown"),
                "page": metadata.get("page", 0),
                "ingested_at": metadata.get("ingested_at"),
            })

        return answer, prioritized_docs, sources


_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service