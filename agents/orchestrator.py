"""
Orchestrator / Router
LangGraph conditional edge function — routes from retriever to correct specialist agents.
Also used as the fan-out router after planner.
"""

from state import AgentState
from typing import List


def route_intent(state: AgentState) -> str:
    """Primary routing: what kind of query is this?"""
    tags = state.get("intent_tags", ["knowledge"])
    # Priority order: ingest > maintenance > compliance > lessons > knowledge
    if "ingest" in tags:
        return "ingest"
    if "maintenance" in tags:
        return "maintenance"
    if "compliance" in tags:
        return "compliance"
    if "lessons" in tags:
        return "lessons"
    return "knowledge"


def route_after_retrieval(state: AgentState) -> List[str]:
    """
    Fan-out: returns the list of agent node names to run in parallel
    based on intent_tags.
    """
    tags = state.get("intent_tags", ["knowledge"])
    targets = []
    tag_map = {
        "knowledge": "knowledge_agent",
        "maintenance": "maintenance_agent",
        "compliance": "compliance_agent",
        "lessons": "lessons_agent",
    }
    for tag in tags:
        node = tag_map.get(tag)
        if node and node not in targets:
            targets.append(node)
    return targets if targets else ["knowledge_agent"]
