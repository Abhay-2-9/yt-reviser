import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag import answer_question, list_categories

app = FastAPI(title="YT Reviser")

# allowed frontends, comma separated (set FRONTEND_ORIGINS on the server)
ORIGINS = [o.strip() for o in os.getenv(
    "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    category: str | None = None   # e.g. "Binary Search"; empty = all topics


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/categories")
def categories():
    return {"categories": list_categories()}


@app.post("/ask")
def ask(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    return answer_question(body.question.strip(), category=body.category or None)