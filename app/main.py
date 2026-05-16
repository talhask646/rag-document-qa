import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.rag_chain import build_rag_chain, ask_question
from app.ingest import ingest_pdf

os.makedirs("uploads", exist_ok=True)

# --- App initialisation ---
app = FastAPI(
    title="RAG Document Q&A",
    description="Upload a PDF and ask questions about it",
    version="1.0.0"
)

# --- CORS ---
# allow_origins=["*"] means any frontend can call this API
# In production you'd restrict this to your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session store ---
# Maps session_id (string) → chain instance (with its own memory)
# This lives in RAM — resets when the server restarts
# Good enough for a portfolio project; production would use Redis
session_chains: dict = {}


# --- Pydantic models ---
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"  # if no session_id sent, use "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str

class UploadResponse(BaseModel):
    message: str
    chunks_stored: int
    filename: str


# --- Helper ---
def get_or_create_chain(session_id: str):
    """
    Returns an existing chain for this session, or builds a new one.
    This ensures each user has their own isolated memory.
    """
    if session_id not in session_chains:
        print(f"Creating new chain for session: {session_id}")
        session_chains[session_id] = build_rag_chain()
    return session_chains[session_id]


# --- Endpoints ---

@app.get("/")
async def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "RAG Document Q&A API is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, runs the ingestion pipeline,
    and stores chunks in Pinecone.
    """
    # Validate file type — reject anything that isn't a PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Validate file size — reject files over 20MB
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:  # 20MB in bytes
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 20MB"
        )

    # Save the uploaded file to the uploads/ directory
    upload_path = os.path.join("uploads", file.filename)
    with open(upload_path, "wb") as f:
        f.write(contents)

    # Run the ingestion pipeline
    try:
        chunks_stored = ingest_pdf(upload_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )

    return UploadResponse(
        message="PDF uploaded and processed successfully",
        chunks_stored=chunks_stored,
        filename=file.filename
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Accepts a question and session_id, runs the RAG chain,
    and returns an answer with source citations.
    """
    # Reject empty questions
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # Get or create the chain for this session
    chain = get_or_create_chain(request.session_id)

    # Run the question through the RAG chain
    try:
        result = ask_question(chain, request.question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clears the memory for a session — used by the 'New Conversation' button.
    """
    if session_id in session_chains:
        del session_chains[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}


# --- Serve frontend ---
# Mounts the frontend/ folder so index.html is accessible at /app
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")