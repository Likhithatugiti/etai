"""
Ingestion Agent — LangGraph node wrapper for document ingestion.
Used when intent_tag == "ingest".
"""

from typing import Callable
from state import AgentState
from ingestion import ingest_documents
from graph import IndustrialKG
from retrieval import HybridRetriever


def make_ingestion_agent(kg: IndustrialKG, retriever: HybridRetriever) -> Callable:
    def ingestion_agent(state: AgentState) -> AgentState:
        file_paths = state.get("entities", {}).get("file_paths", [])
        if not file_paths:
            return {**state, "final_answer": "No file paths provided for ingestion.",
                    "agent_answers": []}

        chunks = ingest_documents(file_paths)

        for chunk in chunks:
            kg.add_chunk(chunk)

        retriever.build(list(retriever._chunks) + chunks if retriever._chunks else chunks)

        return {
            **state,
            "agent_answers": [{
                "agent": "ingestion",
                "sub_question": "ingest",
                "answer": f"Ingested {len(chunks)} chunks from {len(file_paths)} file(s). KG stats: {kg.stats()}",
                "confidence": 1.0,
                "citations": [],
            }],
            "final_answer": f"Successfully ingested {len(chunks)} chunks.",
        }

    return ingestion_agent
