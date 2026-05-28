from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
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
        "Known categories are ORDER, SHIPPING, CANCEL, INVOICE, PAYMENT, REFUND, FEEDBACK, "
        "CONTACT, ACCOUNT, DELIVERY, and SUBSCRIPTION. "
        "For broad phrases like refund requests, shipping examples, or account distribution, "
        "use the matching category filter. Only use an intent filter when the user names an exact "
        "dataset intent such as complaint, get_refund, edit_account, or cancel_order. "
        "Do not invent intent names like REQUEST. If a count returns 0 because the filter was too "
        "specific, retry with the broader category before giving the final answer. "
        "If the tools do not support a question, say so clearly. Keep answers concise."
    )

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        version="v2",
    )


def _message_text(message: BaseMessage) -> str:
    """Return readable text from a LangChain message."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)


def format_react_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a LangGraph ReAct result into CLI-friendly reasoning output."""
    messages = result.get("messages", [])
    reasoning_steps: list[str] = []
    observations: list[str] = []
    selected_tool = "react_agent"
    answer = "I could not produce an answer for that query."

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            thought = _message_text(message).strip()
            if thought:
                reasoning_steps.append(f"Thought: {thought}")
            else:
                reasoning_steps.append("Thought: choose the next dataset tool to call.")

            for tool_call in message.tool_calls:
                selected_tool = tool_call["name"]
                reasoning_steps.append(
                    f"Action: {tool_call['name']}({tool_call.get('args', {})})"
                )

        elif isinstance(message, ToolMessage):
            observation = _message_text(message)
            observations.append(observation)
            reasoning_steps.append(f"Observation: {observation}")

        elif isinstance(message, AIMessage):
            text = _message_text(message).strip()
            if text:
                answer = text

    return {
        "answer": answer,
        "selected_tool": selected_tool,
        "observations": observations,
        "reasoning_steps": reasoning_steps,
    }


def _history_to_messages(history: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    """Convert saved JSON history to LangChain chat tuples."""
    messages: list[tuple[str, str]] = []
    for message in history or []:
        role = message.get("role")
        content = message.get("content", "")
        if role in {"user", "assistant"}:
            messages.append((role, content))
    return messages


def run_react_agent(
    question: str,
    history: list[dict[str, Any]] | None = None,
    recursion_limit: int = 10,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run the optional LLM ReAct agent for one question."""
    agent = build_react_agent()
    messages = _history_to_messages(history)
    messages.append(("user", question))
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    result = agent.invoke(
        {"messages": messages},
        config=config,
    )
    return format_react_result(result)
