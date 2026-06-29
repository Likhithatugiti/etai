"""
Planner Agent
Decomposes the user query into sub-questions and intent tags.
"""

import json
import re
from state import AgentState
from prompts import PLANNER_PROMPT
from utils.llm import llm_invoke


def planner_agent(state: AgentState) -> AgentState:
    prompt = PLANNER_PROMPT.format(query=state["query"])
    raw = llm_invoke(prompt)

    # extract JSON block robustly
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return {
                **state,
                "sub_questions": parsed.get("sub_questions", [state["query"]]),
                "intent_tags": parsed.get("intent_tags", ["knowledge"]),
                "entities": parsed.get("entities", {}),
                "agent_answers": [],
                "critic_scores": [],
                "iteration_count": state.get("iteration_count", 0),
                "hallucination_flag": False,
                "hallucination_reason": "",
                "low_confidence": False,
            }
        except json.JSONDecodeError:
            pass

    # fallback
    return {
        **state,
        "sub_questions": [state["query"]],
        "intent_tags": ["knowledge"],
        "entities": {},
        "agent_answers": [],
        "critic_scores": [],
        "iteration_count": 0,
        "hallucination_flag": False,
        "hallucination_reason": "",
        "low_confidence": False,
    }
