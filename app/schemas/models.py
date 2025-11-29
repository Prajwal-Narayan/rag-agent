from pydantic import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource.
    """
    datasource: Literal["vector_store", "general_chat"] = Field(
        ...,
        description="Given a user question, choose to route it to the 'vector_store' for RAG or 'general_chat' for greeting/casual conversation."
    )