from fastapi import APIRouter, UploadFile, File, HTTPException
from app.api.ingest_service import IngestService
from app.schemas.ingest import IngestResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.api.chat_service import ChatService


router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    service = IngestService()
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
    try:
        result = await service.process_pdf(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    RAG Chat endpoint. Retrieves context and generates an answer.
    """
    # MOVED INSIDE TRY BLOCK 👇
    try:
        service = ChatService()  # If this fails, we want to know why!
        result = await service.chat(request.message)
        return result
    except Exception as e:
        # Now this will print the actual error message to your browser
        raise HTTPException(status_code=500, detail=str(e))