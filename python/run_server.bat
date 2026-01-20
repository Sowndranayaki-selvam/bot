@echo off
setlocal enabledelayedexpansion

echo Q^&A Module - FastAPI Server
echo ==================================================

REM Check if we're in the python directory
if not exist "qa_module.py" (
    echo Error: Please run this from the 'python' directory
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install fastapi uvicorn pydantic PyPDF2 langchain-text-splitters langchain-community langchain-huggingface langchain-core chromadb
)

echo.
echo Starting FastAPI Server...
echo ==================================================
echo.
echo Server will be available at: http://localhost:8000
echo API Documentation at:       http://localhost:8000/docs
echo.
echo Requirements:
echo - Ollama must be running: ollama serve
echo - PDF file must exist at: ../data/exchange.pdf
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
