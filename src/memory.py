import json
from pathlib import Path


MEMORY_DIR = Path("memory/sessions")

MEMORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_session_path(session_id: str) -> Path:
    return MEMORY_DIR / f"{session_id}.json"


def load_session(session_id: str) -> list[dict]:
    path = get_session_path(session_id)

    if not path.exists():
        return []

    with open(path, "r") as file:
        return json.load(file)


def save_session(
    session_id: str,
    messages: list[dict],
) -> None:
    path = get_session_path(session_id)

    with open(path, "w") as file:
        json.dump(
            messages,
            file,
            indent=2,
        )
        
def get_last_user_topic(history: list[dict]) -> dict | None:
    """
    Extract the last remembered dataset topic from conversation history.
    """
    for message in reversed(history):
        if message["role"] != "user":
            continue

        text = message["content"].lower()

        for category in [
            "refund",
            "shipping",
            "account",
            "order",
            "invoice",
            "payment",
            "feedback",
            "delivery",
            "subscription",
            "cancel",
            "contact",
        ]:
            if category in text:
                return {
                    "category": category.upper(),
                    "question": message["content"],
                }

    return None