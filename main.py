"""
AXON — AI for Industrial Knowledge Intelligence
Unified Asset & Operations Brain (ET AI Hackathon 2026 — PS8)

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or Streamlit UI:
    streamlit run app.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from graph import IndustrialKG
from retrieval import HybridRetriever
from pipeline import build_pipeline
from api.routes import router
from api.context import init_context


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise KG, retriever, pipeline."""
    kg = IndustrialKG()
    retriever = HybridRetriever(kg, top_k=8)
    pipeline = build_pipeline(kg, retriever)
    init_context(kg, retriever, pipeline)
    print("[AXON] KG, retriever, and pipeline initialised.")
    yield
    print("[AXON] Shutdown.")


app = FastAPI(
    title="AXON — Industrial Knowledge Intelligence",
    description=(
        "Unified Asset & Operations Brain. "
        "Multi-agent RAG with LangGraph, hybrid retrieval, KG traversal, "
        "critic loop, hallucination guard, and engineer feedback."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "AXON Industrial Knowledge Intelligence",
        "docs": "/docs",
        "endpoints": ["/ingest", "/query", "/feedback", "/kg/stats", "/health"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
