"""
Synthesizer Agent
Merges answers from all specialist agents into one coherent response with citations.
"""

from state import AgentState
from prompts import SYNTHESIZER_PROMPT
from utils.llm import llm_invoke


def synthesizer_agent(state: AgentState) -> AgentState:
    answers = state.get("agent_answers", [])
    chunks = state.get("retrieved_chunks", [])

    if not answers:
        return {**state, "final_answer": "No answers generated.", "source_citations": []}

    agent_answers_text = "\n\n".join(
        f"[{item['agent'].upper()} AGENT — confidence {item.get('confidence', 0):.2f}]\n"
        f"Sub-question: {item.get('sub_question', '')}\n"
        f"Answer: {item['answer']}"
        for item in answers
    )

    prompt = SYNTHESIZER_PROMPT.format(
        query=state["query"],
        agent_answers=agent_answers_text,
    )
    final_answer = llm_invoke(prompt)

    citations = []
    for item in answers:
        for cid in item.get("citations", []):
            chunk = next((c for c in chunks if c["chunk_id"] == cid), None)
            if chunk:
                citations.append({
                    "chunk_id": cid,
                    "document": chunk.get("document_name", "?"),
                    "page": chunk.get("page", 0),
                    "agent": item["agent"],
                })

    return {
        **state,
        "final_answer": final_answer,
        "source_citations": citations,
    }
