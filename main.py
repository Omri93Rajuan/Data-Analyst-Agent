import argparse
from typing import Any

from src.graph import build_graph
from src.memory import load_session, save_session


def _print_reasoning(result: dict[str, Any]) -> None:
    print("\nReasoning steps:")
    for step in result.get("reasoning_steps", []):
        print(f"- {step}")

    observations = result.get("observations", [])
    if observations:
        print("Observations:")
        for observation in observations:
            print(f"- {observation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitext Customer Service Data Analyst Agent")
    parser.add_argument("--session", default="default", help="Session ID for conversation memory.")
    parser.add_argument("--user", default="default_user", help="User ID for profile memory.")
    args = parser.parse_args()

    session_id = args.session
    user_id = args.user
    history = load_session(session_id)
    graph = build_graph()

    print("Customer Service Data Analyst Agent")
    print(f"Session: {session_id}")
    print(f"User: {user_id}")
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
            result = graph.invoke(
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
