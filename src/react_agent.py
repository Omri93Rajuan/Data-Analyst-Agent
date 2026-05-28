from typing import Any

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.config import BASE_URL, MODEL_NAME, NEBIUS_API_KEY
from src.tools import TOOL_REGISTRY


def build_react_agent():
    """Build a LangGraph ReAct agent over the deterministic dataset tools."""
    if not NEBIUS_API_KEY:
        raise ValueError("NEBIUS_API_KEY is required to run the LLM ReAct agent.")

    llm = ChatOpenAI(
        api_key=NEBIUS_API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=0,
        timeout=30,
        max_retries=1,
    )

    tools = [
        StructuredTool.from_function(
            func=entry["function"],
            name=name,
            description=entry["description"],
            args_schema=entry["schema"],
        )
        for name, entry in TOOL_REGISTRY.items()
    ]

    prompt = (
        "You are a data analyst agent for the Bitext Customer Service dataset. "
        "Use tools for factual answers. Do not answer out-of-scope questions from general knowledge. "
        "If the tools do not support a question, say so clearly. Keep answers concise."
    )

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        version="v2",
    )


def run_react_agent(question: str, recursion_limit: int = 10) -> dict[str, Any]:
    """Run the optional LLM ReAct agent for one question."""
    agent = build_react_agent()
    return agent.invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": recursion_limit},
    )
