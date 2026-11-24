from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from app import config
from app.rag import build_vector_store, answer_question

app = FastAPI(
    title="RAG Document QA API",
    description="Upload PDF documents and ask context-grounded questions.",
    version="1.0.0",
)
# Demo storage: in-memory only. Use Redis/database/object storage for production.
STORES: Dict[str, object] = {}

class AskRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=4000)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/documents/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB limit.")
    try:
        store, chunks = build_vector_store(data)
        STORES[session_id] = store
        return {"message": "Document indexed successfully.", "session_id": session_id,
                "chunks_indexed": chunks}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/ask")
def ask(request: AskRequest):
    store = STORES.get(request.session_id)
    if store is None:
        raise HTTPException(404, "No document found for this session. Upload a PDF first.")
    try:
        return answer_question(store, request.question)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@app.delete("/documents/{session_id}")
def delete_document(session_id: str):
    removed = STORES.pop(session_id, None) is not None
    return {"deleted": removed}
