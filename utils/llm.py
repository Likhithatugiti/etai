"""
LLM factory.
Set LLM_PROVIDER=groq  (default) or LLM_PROVIDER=gemini in .env
"""

import os
from functools import lru_cache
from langchain_core.language_models import BaseChatModel


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
        )

    # default: groq
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


def llm_invoke(prompt: str) -> str:
    from langchain_core.messages import HumanMessage
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
