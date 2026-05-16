# RAG Document Q&A System

A production-ready AI-powered document question-answering system built with 
LangChain, Groq (Llama 3.3 70B), Pinecone, and FastAPI.

## Live Demo
🔗 https://web-production-d721d.up.railway.app/app

## What It Does
- Upload any PDF document
- Ask questions in plain English and get accurate answers
- Answers include source citations (exact page numbers)
- Multi-turn conversation memory within a session
- Fully deployed and publicly accessible

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector Database | Pinecone |
| Orchestration | LangChain |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Railway |

## Architecture
1. **Ingestion** — PDF is loaded, split into overlapping chunks, 
   embedded into vectors, and stored in Pinecone
2. **Retrieval** — User question is embedded and matched against 
   stored vectors using MMR similarity search
3. **Generation** — Top chunks are passed to Llama 3.3 70B as 
   context, which generates a grounded answer with citations

## API Endpoints
- `POST /upload` — Upload and process a PDF
- `POST /chat` — Ask a question with session memory
- `DELETE /session/{id}` — Clear conversation memory

## Running Locally
```bash
git clone https://github.com/YOUR_USERNAME/rag-document-qa
cd rag-document-qa
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Add your API keys to .env
uvicorn app.main:app --reload --port 8000
```

## Environment Variables
```
PINECONE_API_KEY=
PINECONE_INDEX_NAME=rag-document-qa
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
```
