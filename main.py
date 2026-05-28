import argparse
from typing import Any

from src.graph import build_graph
from src.memory import load_session, save_session
from src.react_agent import run_react_agent
from src.router import QueryType, route_query


def _print_reasoning(result: dict[str, Any]) -> None:
    print("\nReasoning steps:")
    for step in result.get("reasoning_steps", []):
        print(f"- {step}")

    observations = result.get("observations", [])
    if observations:
        print("Observations:")
        for observation in observations:
            print(f"- {observation}")


def _run_deterministic_graph(
    graph: Any,
    question: str,
    session_id: str,
    user_id: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return graph.invoke(
        {
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
            "history": history,
            "reasoning_steps": [],
        },
        config={
            "recursion_limit": 10,
            "configurable": {"thread_id": session_id},
        },
    )


def _run_react_mode(
    graph: Any,
    question: str,
    session_id: str,
    user_id: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    query_type = route_query(question)
    if query_type in {QueryType.OUT_OF_SCOPE, QueryType.PROFILE}:
        return _run_deterministic_graph(graph, question, session_id, user_id, history)

    try:
        result = run_react_agent(
            question=question,
            history=history,
            recursion_limit=10,
            thread_id=session_id,
        )
        result["query_type"] = query_type.value
        result["reasoning_steps"] = [
            f"router decision: {query_type.value}",
            "mode: llm react agent",
            *result.get("reasoning_steps", []),
            f"selected handler/tool: {result.get('selected_tool', 'react_agent')}",
        ]
        return result
    except Exception as exc:
        result = _run_deterministic_graph(graph, question, session_id, user_id, history)
        result["reasoning_steps"] = [
            "mode: llm react agent unavailable; falling back to deterministic graph",
            f"fallback reason: {exc}",
            *result.get("reasoning_steps", []),
        ]
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitext Customer Service Data Analyst Agent")
    parser.add_argument("--session", default="default", help="Session ID for conversation memory.")
    parser.add_argument("--user", default="default_user", help="User ID for profile memory.")
    parser.add_argument(
        "--mode",
        choices=["deterministic", "react"],
        default="deterministic",
        help="Use deterministic graph mode or the LLM ReAct agent with deterministic fallback.",
    )
    args = parser.parse_args()

    session_id = args.session
    user_id = args.user
    mode = args.mode
    history = load_session(session_id)
    graph = build_graph()

    print("Customer Service Data Analyst Agent")
    print(f"Session: {session_id}")
    print(f"User: {user_id}")
    print(f"Mode: {mode}")
    print("Type 'exit' to quit.\n")

    if history:
        print(f"Loaded {len(history)} messages from session memory.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        try:
            if mode == "react":
                result = _run_react_mode(graph, question, session_id, user_id, history)
            else:
                result = _run_deterministic_graph(graph, question, session_id, user_id, history)
        except Exception as exc:
            print(f"Agent: Sorry, I could not process that query: {exc}\n")
            continue

        _print_reasoning(result)
        answer = result.get("answer", "I could not produce an answer for that query.")
        print(f"\nAgent: {answer}\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        save_session(session_id, history)


if __name__ == "__main__":
    main()
