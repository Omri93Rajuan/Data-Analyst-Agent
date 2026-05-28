from src.tools import count_rows
from src.tools import show_examples


def main() -> None:
    print("\nRefund count")
    print(count_rows(category="REFUND"))

    print("\nCancel order examples")
    print(show_examples(intent="cancel_order", limit=2))


if __name__ == "__main__":
    main()