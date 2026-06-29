"""
Maintenance Agent — RCA, FMEA, maintenance recommendations.
"""

from state import AgentState
from prompts import MAINTENANCE_PROMPT
from utils.llm import llm_invoke


def _format_context(chunks):
    return "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in chunks
    )


def maintenance_agent(state: AgentState) -> AgentState:
    if "maintenance" not in state.get("intent_tags", []):
        return {**state, "agent_answers": []}

    chunks = state.get("retrieved_chunks", [])
    context = _format_context(chunks)
    answers = []

    for sq in state.get("sub_questions", [state["query"]]):
        prompt = MAINTENANCE_PROMPT.format(sub_question=sq, context=context)
        answer = llm_invoke(prompt)
        answers.append({
            "agent": "maintenance",
            "sub_question": sq,
            "answer": answer,
            "confidence": 0.0,
            "citations": [c["chunk_id"] for c in chunks[:3]],
        })

    return {**state, "agent_answers": answers}
