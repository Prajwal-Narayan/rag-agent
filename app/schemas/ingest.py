from pydantic import BaseModel

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_processed: int