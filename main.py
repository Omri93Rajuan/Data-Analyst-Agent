import argparse

from src.memory import load_session
from src.memory import save_session

from src.query_handler import (
    handle_structured_query,
    handle_unstructured_query,
)

from src.router import QueryType
from src.router import route_query


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--session",
        default="default",
        help="Session ID",
    )

    args = parser.parse_args()

    session_id = args.session

    history = load_session(session_id)

    print(
        "Customer Service Data Analyst Agent"
    )

    print(
        f"Session: {session_id}"
    )

    print("Type 'exit' to quit.\n")

    if history:
        print(
            f"Loaded {len(history)} "
            f"messages from memory.\n"
        )

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        query_type = route_query(question)

        print(
            f"\n[Router] "
            f"{query_type.value}"
        )

        if query_type == QueryType.OUT_OF_SCOPE:
            answer = (
                "I can only answer questions "
                "about the Bitext dataset."
            )

        elif (
            query_type
            == QueryType.STRUCTURED
        ):
            answer = handle_structured_query(
                question
            )

        else:
            answer = handle_unstructured_query(
                question
            )

        print(f"Agent: {answer}\n")

        history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        save_session(
            session_id,
            history,
        )


if __name__ == "__main__":
    main()