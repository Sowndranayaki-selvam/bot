"""
FastAPI server to host the Q&A module locally.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qa_module import CustomQAModule
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Q&A Module API",
    description="Local Q&A service based on PDF data",
    version="1.0.0"
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global QA module instance
qa_module = None


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list


class BatchQuestionsRequest(BaseModel):
    questions: list[str]


class BatchQuestionsResponse(BaseModel):
    results: list[QuestionResponse]


@app.on_event("startup")
async def startup_event():
    """Initialize the Q&A module when the server starts."""
    global qa_module
    
    # Adjust path relative to the python directory
    pdf_path = "data/exchange.pdf"
    
    if not os.path.exists(pdf_path):
        # Try alternative path
        pdf_path = "../data/exchange.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF not found at {pdf_path}")
        print("The API will start but questions cannot be answered without a PDF.")
        return
    
    try:
        print("Initializing Q&A module...")
        qa_module = CustomQAModule(pdf_path)
        print("Q&A module ready!")
    except Exception as e:
        print(f"Error initializing Q&A module: {e}")
        print("Make sure Ollama is running and the PDF exists.")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Q&A Module API is running",
        "endpoints": {
            "ask": "/ask (POST) - Ask a single question",
            "ask_batch": "/ask_batch (POST) - Ask multiple questions",
            "health": "/health (GET) - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "qa_module_loaded": qa_module is not None
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a single question to the Q&A module.
    
    Args:
        request: QuestionRequest with 'question' field
        
    Returns:
        QuestionResponse with 'answer' and 'sources' fields
    """
    if qa_module is None:
        raise HTTPException(
            status_code=503,
            detail="Q&A module not initialized. Make sure the PDF exists and Ollama is running."
        )
    
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        result = qa_module.ask(request.question)
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/ask_batch", response_model=BatchQuestionsResponse)
async def ask_batch(request: BatchQuestionsRequest):
    """
    Ask multiple questions to the Q&A module.
    
    Args:
        request: BatchQuestionsRequest with 'questions' list
        
    Returns:
        BatchQuestionsResponse with list of QuestionResponse objects
    """
    if qa_module is None:
        raise HTTPException(
            status_code=503,
            detail="Q&A module not initialized. Make sure the PDF exists and Ollama is running."
        )
    
    if not request.questions or len(request.questions) == 0:
        raise HTTPException(
            status_code=400,
            detail="Questions list cannot be empty"
        )
    
    try:
        results = []
        for question in request.questions:
            if question.strip():
                result = qa_module.ask(question)
                results.append(QuestionResponse(
                    answer=result["answer"],
                    sources=result["sources"]
                ))
        
        return BatchQuestionsResponse(results=results)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing questions: {str(e)}"
        )


if __name__ == "__main__":
    print("Starting Q&A Module FastAPI Server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("\nMake sure Ollama is running (ollama serve) before asking questions!")
    
    uvicorn.run(
    app,
    host="127.0.0.1",
    port=8000,
    log_level="info"
)

