"""
Knowledge Agent — answers general engineering queries from the KG + RAG context.
"""

from state import AgentState
from prompts import KNOWLEDGE_PROMPT
from utils.llm import llm_invoke


def _format_context(chunks):
    return "\n\n".join(
        f"[{c['chunk_id']}] ({c.get('document_name','?')}, p.{c.get('page',0)})\n{c['text']}"
        for c in chunks
    )


def knowledge_agent(state: AgentState) -> AgentState:
    chunks = state.get("retrieved_chunks", [])
    context = _format_context(chunks)
    answers = []

    for sq in state.get("sub_questions", [state["query"]]):
        if "knowledge" not in state.get("intent_tags", ["knowledge"]):
            continue
        prompt = KNOWLEDGE_PROMPT.format(sub_question=sq, context=context)
        answer = llm_invoke(prompt)
        answers.append({
            "agent": "knowledge",
            "sub_question": sq,
            "answer": answer,
            "confidence": 0.0,  # filled by critic
            "citations": [c["chunk_id"] for c in chunks[:3]],
        })

    return {**state, "agent_answers": answers}
