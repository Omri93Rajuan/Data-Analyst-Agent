from src.query_handler import handle_structured_query, handle_unstructured_query
from src.router import QueryType, route_query


def main() -> None:
    print("Customer Service Data Analyst Agent")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        query_type = route_query(question)
        print(f"\n[Router] {query_type.value}")

        if query_type == QueryType.OUT_OF_SCOPE:
            print("Agent: I can only answer questions about the Bitext dataset.\n")
            continue

        if query_type == QueryType.STRUCTURED:
            answer = handle_structured_query(question)
            print(f"Agent: {answer}\n")
            continue

        if query_type == QueryType.UNSTRUCTURED:
            answer = handle_unstructured_query(question)
            print(f"Agent: {answer}\n")
            continue


if __name__ == "__main__":
    main()