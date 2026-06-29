from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str
    session_id: str
    user_id: str

    # ── Planner output ─────────────────────────────────────────────────────
    sub_questions: List[str]          # decomposed sub-queries
    intent_tags: List[str]            # e.g. ["maintenance", "compliance"]
    entities: Dict[str, Any]          # {equipment_tags, reg_refs, concepts}

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieved_chunks: List[Dict]      # [{text, doc_id, page, score, source}]
    graph_context: Dict[str, Any]     # KG subgraph: nodes + typed edges

    # ── Specialist agent answers (additive — parallel fan-in) ──────────────
    agent_answers: Annotated[List[Dict], operator.add]
    # each: {agent: str, answer: str, confidence: float, citations: List}

    # ── Critic ─────────────────────────────────────────────────────────────
    critic_scores: List[float]
    low_confidence: bool
    iteration_count: int              # guard against infinite retry loops

    # ── Synthesis + guard ──────────────────────────────────────────────────
    final_answer: str
    source_citations: List[Dict]
    hallucination_flag: bool
    hallucination_reason: str

    # ── Feedback / update ──────────────────────────────────────────────────
    feedback: Optional[str]
    correction_text: Optional[str]
