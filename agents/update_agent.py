"""
Update Agent
Processes engineer feedback: ingests correction into KG and rebuilds retrieval index.
"""

from typing import Callable
from state import AgentState
from graph import IndustrialKG
from retrieval import HybridRetriever
from ingestion import ingest_documents
import tempfile, os


def make_update_agent(kg: IndustrialKG, retriever: HybridRetriever) -> Callable:
    def update_agent(state: AgentState) -> AgentState:
        correction = state.get("correction_text", "")
        if not correction:
            return state

        # Write correction as a temp file and ingest it
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False, encoding="utf-8") as f:
            f.write(correction)
            tmp_path = f.name

        new_chunks = ingest_documents([tmp_path])
        os.unlink(tmp_path)

        # Update KG
        kg.update_from_correction(correction)

        # Rebuild retrieval index with new chunks
        retriever.update(new_chunks)

        return {
            **state,
            "final_answer": (
                f"Knowledge base updated with {len(new_chunks)} new chunk(s) from engineer feedback. "
                f"KG now has {kg.stats()['nodes']} nodes and {kg.stats()['edges']} edges."
            ),
        }

    return update_agent
