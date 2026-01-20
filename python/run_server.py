#!/usr/bin/env python
"""
Run the FastAPI server with dependency checks.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "PyPDF2",
        "langchain_text_splitters",
        "langchain_community",
        "langchain_huggingface",
        "langchain_core",
        "langchain_ollama",
        "chromadb"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("\nInstalling missing packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "fastapi", "uvicorn", "pydantic",
            "PyPDF2", "langchain-text-splitters",
            "langchain-community", "langchain-huggingface",
            "langchain-core", "langchain-ollama", "chromadb"
        ])
    else:
        print("✓ All dependencies are installed")

def main():
    """Main entry point."""
    print("Q&A Module - FastAPI Server")
    print("=" * 50)
    
    # Check dependencies
    print("\nChecking dependencies...")
    check_dependencies()
    
    # Check if running from correct directory
    if not os.path.exists("qa_module.py"):
        print("\nError: Please run this script from the 'python' directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Import and run the server
    print("\nStarting server...")
    print("=" * 50)
    
    try:
        from main import app
        import uvicorn
        
        print("Server starting at http://localhost:8000")
        print("Interactive docs at http://localhost:8000/docs")
        print("Swagger UI at http://localhost:8000/swagger-ui")
        print("\nPress Ctrl+C to stop the server\n")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
