"""
Hallucination Guard
Checks the synthesized answer against retrieved chunks for unsupported specific claims.
"""

import json
import re
from state import AgentState
from prompts import HALLUCINATION_GUARD_PROMPT
from utils.llm import llm_invoke


def hallucination_guard(state: AgentState) -> AgentState:
    answer = state.get("final_answer", "")
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(f"[{c['chunk_id']}] {c['text'][:300]}" for c in chunks[:6])

    prompt = HALLUCINATION_GUARD_PROMPT.format(answer=answer, context=context)
    raw = llm_invoke(prompt)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    hallucination = False
    reason = ""
    if match:
        try:
            parsed = json.loads(match.group())
            hallucination = bool(parsed.get("hallucination_detected", False))
            reason = parsed.get("reason", "")
        except json.JSONDecodeError:
            pass

    return {
        **state,
        "hallucination_flag": hallucination,
        "hallucination_reason": reason,
    }


def check_hallucination(state: AgentState) -> str:
    """LangGraph conditional edge."""
    if state.get("hallucination_flag", False):
        return "flag"
    return "clear"
