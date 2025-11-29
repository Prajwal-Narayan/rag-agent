from fastapi import APIRouter, UploadFile, File, HTTPException
from app.api.ingest_service import IngestService
from app.schemas.ingest import IngestResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.api.chat_service import ChatService
from typing import List, Dict, Any


router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    service = IngestService()

    filename = file.filename or ""
    
    if not filename.lower().endswith(".pdf"):
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
    
@router.post("/eval", response_model=Dict[str, Any])
async def run_evaluation():
    """
    Runs a live benchmark on the RAG system using Ragas.
    """
    # Define 3 hard questions to stress-test your PDF
    test_questions = [
        "Summarize the document.",
        "What are the limitations mentioned?",
        "How is the training process described?"
    ]
    
    from app.evaluation.ragas_eval import RagasEvaluator
    
    try:
        evaluator = RagasEvaluator()
        results = await evaluator.run_benchmark(test_questions)
        
        # Return the scores
        return {
            "success": True,
            "metrics": results,
            "questions_tested": test_questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))