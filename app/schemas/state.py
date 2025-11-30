from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: The user's question.
        generation: The LLM's answer.
        documents: List of retrieved documents.
        messages: Chat history (automatically handles appending new messages).
    """
    question: str
    generation: str
    documents: List[str]
    # 'add_messages' tells LangGraph to append new messages to history
    # instead of overwriting the whole list.
    messages: Annotated[List[BaseMessage], add_messages]