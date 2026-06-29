"""
Lessons Learned Agent — past incidents, systemic failure patterns, proactive warnings.
"""

from state import AgentState
from prompts import LESSONS_PROMPT
from utils.llm import llm_invoke


def _format_context(chunks):
    return "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in chunks
    )


def lessons_agent(state: AgentState) -> AgentState:
    if "lessons" not in state.get("intent_tags", []):
        return {**state, "agent_answers": []}

    chunks = state.get("retrieved_chunks", [])
    context = _format_context(chunks)
    answers = []

    for sq in state.get("sub_questions", [state["query"]]):
        prompt = LESSONS_PROMPT.format(sub_question=sq, context=context)
        answer = llm_invoke(prompt)
        answers.append({
            "agent": "lessons",
            "sub_question": sq,
            "answer": answer,
            "confidence": 0.0,
            "citations": [c["chunk_id"] for c in chunks[:3]],
        })

    return {**state, "agent_answers": answers}
