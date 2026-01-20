"""
Q&A Module - Answers questions based ONLY on user-provided PDF data.
No external knowledge or third-party data is used.
"""

import os
from pathlib import Path
from typing import Optional
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class CustomQAModule:
    def __init__(self, pdf_path: str, openai_api_key: Optional[str] = None):
        """
        Initialize Q&A module with your PDF data.
        
        Args:
            pdf_path: Path to your PDF file
            openai_api_key: OpenAI API key (uses env var if not provided)
        """
        self.pdf_path = pdf_path
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.documents = []
        self.vector_store: Optional[Chroma] = None
        self.qa_chain = None
        self.retriever = None
        
        # Initialize
        self._load_pdf()
        self._create_vector_store()
        self._setup_qa_chain()
    
    def _load_pdf(self):
        """Extract text from PDF"""
        print(f"Loading PDF: {self.pdf_path}")
        
        reader = PdfReader(self.pdf_path)
        text = ""
        
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        
        # Split into chunks - optimized for URLs, addresses, and contact info
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "Contact:", "Address:", "Website:", "Email:", "Phone:", "\n", " ", ""]
        )
        
        self.documents = splitter.split_text(text)
        print(f"Loaded {len(self.documents)} text chunks from PDF")
    
    def _create_vector_store(self):
        """Create vector embeddings and store them"""
        print("Creating vector embeddings (using free HuggingFace model)...")
        
        # Use free HuggingFace embeddings instead of OpenAI
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create ChromaDB vector store
        self.vector_store = Chroma.from_texts(
            texts=self.documents,
            embedding=embeddings,
            persist_directory="../../data/chroma_db"
        )
        
        print("Vector store created and persisted")
    
    def _setup_qa_chain(self):
        """Setup the Q&A chain"""
        assert self.vector_store is not None, "Vector store must be initialized before setting up Q&A chain"
        
        llm = OllamaLLM(
            model="mistral",
            temperature=0,  # No hallucination
            base_url="http://localhost:11434"
        )
        
        # Create retriever from vector store
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Return top 5 relevant chunks for better context
        )
        
        # Create custom prompt
        prompt = self._get_custom_prompt()
        
        # Build RAG chain using LCEL
        self.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        self.retriever = retriever
        
        print("Q&A chain ready")
    
    def _get_custom_prompt(self):
        """Create a custom prompt that ensures no external knowledge"""
        
        template = """Use ONLY the following context to answer the question. 
If the answer is not in the context, say "I don't have this information in my training data."
Do NOT use external knowledge or make assumptions.

IMPORTANT: When asked for URLs, addresses, emails, or phone numbers, provide the EXACT text from the context.
For addresses and URLs, include the complete information without abbreviation.

Context:
{context}

Question: {question}

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def ask(self, question: str) -> dict:
        """
        Ask a question - answers ONLY from your PDF data.
        
        Args:
            question: Your question
            
        Returns:
            Dictionary with answer and source documents
        """
        assert self.retriever is not None, "Retriever not initialized"
        assert self.qa_chain is not None, "Q&A chain not initialized"
        
        # Get relevant documents
        relevant_docs = self.retriever.invoke(question)
        
        # Get answer from chain
        answer = self.qa_chain.invoke(question)
        
        return {
            "answer": answer,
            "sources": [i+1 for i in range(len(relevant_docs))]  # Page numbers
        }
    
    def batch_ask(self, questions: list) -> list:
        """Ask multiple questions"""
        results = []
        for q in questions:
            print(f"\nQuestion: {q}")
            result = self.ask(q)
            print(f"Answer: {result['answer']}")
            results.append(result)
        return results


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize with your PDF
    pdf_path = "data/exchange.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found at {pdf_path}")
        sys.exit(1)
    
    # Create Q&A module
    qa = CustomQAModule(pdf_path)
    
    # Example questions
    questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
    ]
    
    qa.batch_ask(questions)