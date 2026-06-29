"""
AXON Pipeline — LangGraph StateGraph

Flow:
  planner → retriever → [knowledge | maintenance | compliance | lessons] (parallel)
          → critic → (retry → retriever | proceed → synthesizer)
          → hallucination_guard → (flag → synthesizer | clear → END)

Ingestion and update are separate entry points via /ingest and /feedback API routes.
"""

from langgraph.graph import StateGraph, END
from state import AgentState
from graph import IndustrialKG
from retrieval import HybridRetriever

from agents import (
    planner_agent,
    knowledge_agent, maintenance_agent, compliance_agent, lessons_agent,
    critic_agent, check_confidence,
    synthesizer_agent,
    hallucination_guard, check_hallucination,
    make_retriever_node, make_ingestion_agent, make_update_agent,
)


def build_pipeline(kg: IndustrialKG, retriever: HybridRetriever):
    """Build and compile the AXON LangGraph pipeline."""

    retriever_node = make_retriever_node(retriever, kg)
    ingestion_agent = make_ingestion_agent(kg, retriever)
    update_agent = make_update_agent(kg, retriever)

    workflow = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    workflow.add_node("planner", planner_agent)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("knowledge_agent", knowledge_agent)
    workflow.add_node("maintenance_agent", maintenance_agent)
    workflow.add_node("compliance_agent", compliance_agent)
    workflow.add_node("lessons_agent", lessons_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("synthesizer", synthesizer_agent)
    workflow.add_node("hallucination_guard", hallucination_guard)
    workflow.add_node("ingestion_agent", ingestion_agent)
    workflow.add_node("update_agent", update_agent)

    # ── Entry point ──────────────────────────────────────────────────────────
    workflow.set_entry_point("planner")

    # Planner → route (ingest goes straight to ingestion_agent)
    workflow.add_conditional_edges(
        "planner",
        lambda s: "ingest" if "ingest" in s.get("intent_tags", []) else "retrieve",
        {"ingest": "ingestion_agent", "retrieve": "retriever"},
    )

    # Retriever → fan-out to specialist agents based on intent_tags
    workflow.add_conditional_edges(
        "retriever",
        lambda s: s.get("intent_tags", ["knowledge"])[0],
        {
            "knowledge": "knowledge_agent",
            "maintenance": "maintenance_agent",
            "compliance": "compliance_agent",
            "lessons": "lessons_agent",
        },
    )
    # All non-primary tags also run (parallel: all agents → critic)
    workflow.add_edge("knowledge_agent", "critic")
    workflow.add_edge("maintenance_agent", "critic")
    workflow.add_edge("compliance_agent", "critic")
    workflow.add_edge("lessons_agent", "critic")

    # Critic → retry or proceed
    workflow.add_conditional_edges(
        "critic", check_confidence,
        {"retry": "retriever", "proceed": "synthesizer"},
    )

    # Synthesizer → hallucination guard
    workflow.add_edge("synthesizer", "hallucination_guard")

    # Hallucination guard → re-synthesize or done
    workflow.add_conditional_edges(
        "hallucination_guard", check_hallucination,
        {"flag": "synthesizer", "clear": END},
    )

    # Terminal edges
    workflow.add_edge("ingestion_agent", END)
    workflow.add_edge("update_agent", END)

    return workflow.compile()
