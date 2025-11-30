from pydantic import BaseModel
from typing import List

# class ChatRequest(BaseModel):
#     message: str # user query

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]     # output response with sources

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "user_123" # Default ID for testing

    