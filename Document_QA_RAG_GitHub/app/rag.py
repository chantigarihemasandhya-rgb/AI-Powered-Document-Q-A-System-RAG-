from __future__ import annotations

import io
from typing import List
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app import config

SYSTEM_PROMPT = """You answer questions using only the supplied document context.
If the answer is not present in the context, say that the document does not
provide enough information. Do not invent facts. When possible, mention page
numbers that support the answer.

DOCUMENT CONTEXT:
{context}
"""

def extract_pdf(file_bytes: bytes) -> List[Document]:
    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"page": page_num}
            ))
    if not docs:
        raise ValueError("No extractable text found. The PDF may be scanned; run OCR first.")
    return docs

def build_vector_store(file_bytes: bytes) -> tuple[FAISS, int]:
    if config.CHUNK_OVERLAP >= config.CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    pages = extract_pdf(file_bytes)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("The PDF did not produce any searchable text chunks.")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings), len(chunks)

def answer_question(vector_store: FAISS, question: str) -> dict:
    if not question.strip():
        raise ValueError("Please enter a question.")
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    matches = vector_store.similarity_search(question, k=config.TOP_K)
    context_parts = []
    sources = []
    for doc in matches:
        page = doc.metadata.get("page", "unknown")
        context_parts.append(f"[Page {page}]\n{doc.page_content}")
        sources.append({"page": page, "excerpt": doc.page_content[:350]})
    context = "\n\n---\n\n".join(context_parts)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nAnswer clearly and concisely.")
    ])
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        api_key=config.OPENAI_API_KEY,
    )
    response = (prompt | llm).invoke({"context": context, "question": question})
    return {"answer": response.content, "sources": sources}
