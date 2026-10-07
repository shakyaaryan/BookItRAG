from langchain_core.messages import BaseMessage


def extract_llm_text(response: BaseMessage) -> str:
    """Extract text from LLM response, handling both string and dict content blocks.
    
    Gemini returns content as list of dicts: [{'type': 'text', 'text': '...', 'extras': {...}}]
    The BaseMessage.text property handles this correctly.
    """
    return response.text if response.text else ""