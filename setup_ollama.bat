@echo off
REM Quick Ollama setup for Windows
echo.
echo ============================================================
echo 🤖 Setting up Ollama (Local AI - No API Key Needed!)
echo ============================================================
echo.
echo This script will:
echo 1. Download Ollama (if not installed)
echo 2. Install Mistral model
echo 3. Start Ollama server
echo.
echo Open https://ollama.ai to install Ollama manually if needed
echo.

REM Check if ollama is installed
ollama --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Ollama not found. Please install it from https://ollama.ai
    echo Then run this script again.
    pause
    exit /b 1
)

echo ✅ Ollama found

REM Check if mistral model is installed
echo.
echo Checking for Mistral model...
ollama list | find "mistral" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 📥 Downloading Mistral model (first time, ~5-10 min)...
    call ollama pull mistral
)

echo.
echo ✅ Mistral model ready
echo.
echo 🚀 Starting Ollama server...
echo The model will now be available on http://localhost:11434
echo Press Ctrl+C to stop the server when done.
echo.
call ollama serve
