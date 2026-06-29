"""
FastAPI routes for AXON.

POST /ingest       — upload documents, ingest into KG + vector store
POST /query        — run the AXON LangGraph pipeline
POST /feedback     — engineer correction → update KG + retriever
GET  /kg/stats     — KG node/edge statistics
GET  /health       — liveness check
"""

import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

# ── Shared singletons (initialised in main.py, injected here) ────────────────
from api.context import get_kg, get_retriever, get_pipeline
from ingestion import ingest_documents

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Request / response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"


class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    correction_text: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "service": "AXON Industrial Knowledge Intelligence"}


@router.get("/kg/stats")
async def kg_stats():
    kg = get_kg()
    return kg.stats()


@router.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    kg = get_kg()
    retriever = get_retriever()
    saved = []

    for f in files:
        dest = os.path.join(UPLOAD_DIR, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)

    chunks = ingest_documents(saved)
    for chunk in chunks:
        kg.add_chunk(chunk)

    existing = retriever._chunks if retriever._chunks else []
    retriever.build(existing + chunks)

    return {
        "files_ingested": len(saved),
        "chunks_created": len(chunks),
        "kg_stats": kg.stats(),
    }


@router.post("/query")
async def query(req: QueryRequest):
    pipeline = get_pipeline()
    initial_state = {
        "query": req.query,
        "session_id": req.session_id or str(uuid.uuid4()),
        "user_id": req.user_id or "anonymous",
        "sub_questions": [],
        "intent_tags": [],
        "entities": {},
        "retrieved_chunks": [],
        "graph_context": {},
        "agent_answers": [],
        "critic_scores": [],
        "low_confidence": False,
        "iteration_count": 0,
        "final_answer": "",
        "source_citations": [],
        "hallucination_flag": False,
        "hallucination_reason": "",
        "feedback": None,
        "correction_text": None,
    }
    result = pipeline.invoke(initial_state)
    return {
        "query": req.query,
        "intent_tags": result.get("intent_tags", []),
        "sub_questions": result.get("sub_questions", []),
        "final_answer": result.get("final_answer", ""),
        "source_citations": result.get("source_citations", []),
        "hallucination_flag": result.get("hallucination_flag", False),
        "hallucination_reason": result.get("hallucination_reason", ""),
        "critic_scores": result.get("critic_scores", []),
        "agent_answers": result.get("agent_answers", []),
        "kg_nodes_used": len(result.get("graph_context", {}).get("nodes", [])),
    }


@router.post("/feedback")
async def feedback(req: FeedbackRequest):
    kg = get_kg()
    retriever = get_retriever()

    if not req.correction_text.strip():
        raise HTTPException(status_code=400, detail="correction_text is empty")

    # Directly update KG + retriever (no full pipeline needed)
    kg.update_from_correction(req.correction_text)
    from ingestion import ingest_documents
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write(req.correction_text)
        tmp = f.name
    new_chunks = ingest_documents([tmp])
    os.unlink(tmp)
    retriever.update(new_chunks)

    return {
        "status": "updated",
        "new_chunks": len(new_chunks),
        "kg_stats": kg.stats(),
    }
