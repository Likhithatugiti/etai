"""
Retriever Node
Runs hybrid retrieval (BM25 + FAISS + KG traversal) and stores results in state.
Injected with retriever and kg instances via closure.
"""

from typing import Callable
from state import AgentState
from retrieval import HybridRetriever
from graph import IndustrialKG


def make_retriever_node(retriever: HybridRetriever, kg: IndustrialKG) -> Callable:
    def retriever_node(state: AgentState) -> AgentState:
        query = state["query"]
        entities = state.get("entities", {})

        # Retrieve chunks
        chunks = retriever.retrieve(query, entities)

        # KG subgraph for equipment tags
        tags = entities.get("equipment_tags", [])
        seed_nodes = kg.find_equipment_nodes(tags) if tags else []
        graph_context = kg.get_subgraph(seed_nodes, hops=2) if seed_nodes else {"nodes": [], "edges": []}

        return {
            **state,
            "retrieved_chunks": chunks,
            "graph_context": graph_context,
        }

    return retriever_node
