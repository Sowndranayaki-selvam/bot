"""
Q&A Module - Answers questions based ONLY on user-provided PDF data.
No external knowledge or third-party data is used.
"""

import os
from typing import Optional, List
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document



class CustomQAModule:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.documents: List[Document] = []
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.qa_chain = None

        # Initialize pipeline
        self._load_pdf()
        self._create_vector_store()
        self._setup_qa_chain()

    # -------------------------------------------------
    # 1. LOAD AND SPLIT PDF PROPERLY
    # -------------------------------------------------
    def _load_pdf(self):
        print(f"Loading PDF: {self.pdf_path}")

        reader = PdfReader(self.pdf_path)
        docs = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": page_num + 1}
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        self.documents = splitter.split_documents(docs)
        print(f"Loaded {len(self.documents)} chunks from PDF")

    # -------------------------------------------------
    # 2. CREATE / REBUILD VECTOR STORE (CORRECT PATH)
    # -------------------------------------------------
    def _create_vector_store(self):
        print("Creating vector embeddings...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # IMPORTANT: always use correct path
        persist_path = "data/chroma_db"

        # Delete old DB to avoid stale indexes
        if os.path.exists(persist_path):
            print("Removing old vector DB...")
            import shutil
            shutil.rmtree(persist_path)

        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=embeddings,
            persist_directory=persist_path
        )

        self.vector_store.persist()
        print("Vector store created successfully")

    # -------------------------------------------------
    # 3. SETUP RAG CHAIN PROPERLY
    # -------------------------------------------------
    def _setup_qa_chain(self):
        assert self.vector_store is not None

        llm = OllamaLLM(
            model="mistral",
            temperature=0,
            base_url="http://localhost:11434"
        )

        # Increase retrieval depth (deep pages!)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )

        prompt = self._get_custom_prompt()

        def format_docs(docs):
            return "\n\n".join(
                [f"(Page {d.metadata['page']})\n{d.page_content}" for d in docs]
            )

        # Proper RAG pipeline
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": lambda x: x
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("Q&A chain ready")

    # -------------------------------------------------
    # 4. PROMPT
    # -------------------------------------------------
    def _get_custom_prompt(self):
        template = """Use ONLY the following context to answer the question.
If the answer is not present in the context, say:
"I don't have this information in my training data."

Do NOT use any external knowledge.

Context:
{context}

Question: {question}

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    # -------------------------------------------------
    # 5. ASK FUNCTION WITH DEBUG
    # -------------------------------------------------
    def ask(self, question: str) -> dict:
        assert self.retriever is not None
        assert self.qa_chain is not None

        # Retrieve docs (debug)
        relevant_docs = self.retriever.invoke(question)

        print("\n--- RETRIEVED CONTEXT PREVIEW ---")
        for d in relevant_docs[:3]:
            print(f"[Page {d.metadata['page']}] {d.page_content[:300]}...\n")

        # Generate answer
        answer = self.qa_chain.invoke(question)

        return {
            "answer": answer,
            "sources": [d.metadata["page"] for d in relevant_docs]
        }


# -------------------------------------------------
# TEST LOCALLY
# -------------------------------------------------
if __name__ == "__main__":

    pdf_path = "data/exchange.pdf"

    if not os.path.exists(pdf_path):
        print(f"PDF not found at {pdf_path}")
        exit(1)

    qa = CustomQAModule(pdf_path)

    while True:
        q = input("\nAsk a question (or type exit): ")
        if q.lower() == "exit":
            break

        result = qa.ask(q)
        print("\nANSWER:", result["answer"])
        print("SOURCES:", result["sources"])
