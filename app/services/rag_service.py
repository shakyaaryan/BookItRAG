from typing import List, Dict, Any, Optional
from uuid import UUID
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import settings
from app.services.vector_store import get_vector_store
from app.services.memory_service import get_memory_service


class RAGService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
        )
        self.vector_store = get_vector_store()
        self.memory_service = get_memory_service()

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
        return response.content.strip()

    def _retrieve_context(self, query: str, user_id: UUID, top_k: int = 4) -> List[Dict[str, Any]]:
        return self.vector_store.similarity_search(query, user_id, top_k)

    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "No relevant documents found."

        context_parts = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            source = metadata.get("source", "unknown")
            page = metadata.get("page", 0)
            text = metadata.get("text", "")
            context_parts.append(f"[Source: {source}, Page: {page}]\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        user_id: UUID,
        session_id: str,
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        history = self.memory_service.get_history(user_id, session_id)
        standalone_query = self._rephrase_query(query, history)
        retrieved_docs = self._retrieve_context(standalone_query, user_id)
        context = self._format_context(retrieved_docs)

        history_messages = self._format_history(history)

        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain enough information, say so.
Be concise and helpful. Cite sources using [Source: filename, Page: X] format when referencing information."""

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(history_messages)
        messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {standalone_query}"))

        response = self.llm.invoke(messages)
        answer = response.content.strip()

        sources = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            sources.append({
                "file_name": metadata.get("source", "unknown"),
                "page": metadata.get("page", 0),
            })

        return answer, retrieved_docs, sources


_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service