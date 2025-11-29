from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    message: str # user query

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]     # output response with sources

    