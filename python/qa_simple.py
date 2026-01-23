"""
Simplified Q&A Module - Lightweight version without heavy dependencies
"""

import os
from typing import Optional
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class CustomQAModule:
    def __init__(self, pdf_path: str):
        """Initialize Q&A module with your PDF data."""
        self.pdf_path = pdf_path
        self.documents = []
        self.vector_store: Optional[Chroma] = None
        self.qa_chain = None
        self.retriever = None
        
        self._load_pdf()
        self._create_vector_store()
        self._setup_qa_chain()
    
    def _load_pdf(self):
        """Extract text from PDF"""
        print(f" Loading PDF: {self.pdf_path}")
        
        reader = PdfReader(self.pdf_path)
        text = ""
        
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.documents = splitter.split_text(text)
        print(f"Loaded {len(self.documents)} text chunks from PDF")
    
    def _create_vector_store(self):
        """Create vector embeddings using Ollama"""
        print("Creating vector embeddings (using Ollama)...")
        
       
        embeddings = OllamaEmbeddings(
            model="mistral",
            base_url="http://localhost:11434"
        )
        
       
        self.vector_store = Chroma.from_texts(
            texts=self.documents,
            embedding=embeddings,
            persist_directory="../../data/chroma_db"
        )
        
        print("Vector store created")
    
    def _setup_qa_chain(self):
        """Setup the Q&A chain"""
        assert self.vector_store is not None, "Vector store must be initialized before setting up Q&A chain"
        
        print("Setting up Q&A chain...")
        
        llm = Ollama(
            model="mistral",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        prompt = PromptTemplate(
            template="""Use ONLY the following context to answer the question. 
If the answer is not in the context, say "I don't have this information in my training data."
Do NOT use external knowledge or make assumptions.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        self.retriever = retriever
        
        print("Q&A chain ready!")
    
    def ask(self, question: str) -> dict:
        """Ask a question - answers ONLY from your PDF data."""
        assert self.retriever is not None, "Retriever not initialized"
        assert self.qa_chain is not None, "Q&A chain not initialized"
        
        relevant_docs = self.retriever.invoke(question)
        answer = self.qa_chain.invoke(question)
        
        return {
            "answer": answer,
            "sources": [i+1 for i in range(len(relevant_docs))]
        }
    
    def batch_ask(self, questions: list) -> list:
        """Ask multiple questions"""
        results = []
        for q in questions:
            print(f"\nQuestion: {q}")
            result = self.ask(q)
            print(f"Answer: {result['answer']}\n")
            results.append(result)
        return results