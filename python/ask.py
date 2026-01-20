"""
Simple interface to use the Q&A module.
Just ask questions and get answers from your data!
"""

import os
import sys
import subprocess
from qa_module import CustomQAModule


def main():
    print("=" * 60)
    print(" Q&A Module - Trained on Your PDF Data Only")
    print("=" * 60)
    
    # Check for Ollama (local LLM - no API key needed!)
    print("\n  Checking for Ollama (local AI model)...")
    try:
        response = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            timeout=2
        )
        if response.returncode != 0:
            raise Exception("Ollama not found")
        print(" Ollama found")
    except Exception as e:
        print(f" Ollama not installed or not running")
        print("   1. Install from: https://ollama.ai")
        print("   2. Run: ollama pull mistral")
        print("   3. Run: ollama serve")
        print("   4. Then run this script again")
        return
    
    # Check for PDF - adjust path to go up to data folder
    pdf_path = "../data/exchange.pdf"
    if not os.path.exists(pdf_path):
        print(f" Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    # Initialize module
    print("\n Initializing Q&A module...")
    qa = CustomQAModule(pdf_path)
    
    # Interactive Q&A
    print("\n" + "=" * 60)
    print(" Ask questions about your PDF (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        question = input("\n Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n Searching your data...")
        result = qa.ask(question)
        
        print(f"\n Answer:\n{result['answer']}")
        if result['sources']:
            print(f"\n Source pages: {result['sources']}")


if __name__ == "__main__":
    main()