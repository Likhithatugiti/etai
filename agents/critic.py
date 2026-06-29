"""
Critic Agent
Scores each agent answer on [0,1]. Sets low_confidence=True if average < threshold.
Max 2 retry iterations to prevent infinite loops.
"""

import json
import re
from state import AgentState
from prompts import CRITIC_PROMPT
from utils.llm import llm_invoke

CONFIDENCE_THRESHOLD = 0.60
MAX_ITERATIONS = 2


def critic_agent(state: AgentState) -> AgentState:
    answers = state.get("agent_answers", [])
    chunks = state.get("retrieved_chunks", [])
    chunk_ids = [c["chunk_id"] for c in chunks[:5]]
    scores = []

    for item in answers:
        prompt = CRITIC_PROMPT.format(
            sub_question=item.get("sub_question", ""),
            answer=item.get("answer", ""),
            chunk_ids=", ".join(chunk_ids),
        )
        raw = llm_invoke(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        score = 0.5
        if match:
            try:
                parsed = json.loads(match.group())
                score = float(parsed.get("score", 0.5))
                item["confidence"] = score
                item["critic_reason"] = parsed.get("reason", "")
            except (json.JSONDecodeError, ValueError):
                item["confidence"] = score
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    iteration = state.get("iteration_count", 0)
    low_conf = avg_score < CONFIDENCE_THRESHOLD and iteration < MAX_ITERATIONS

    return {
        **state,
        "critic_scores": scores,
        "low_confidence": low_conf,
        "iteration_count": iteration + 1,
        "agent_answers": answers,
    }


def check_confidence(state: AgentState) -> str:
    """LangGraph conditional edge: retry retrieval or proceed to synthesis."""
    if state.get("low_confidence", False):
        return "retry"
    return "proceed"
