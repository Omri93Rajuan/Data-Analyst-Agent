import json
import re
from pathlib import Path
from typing import Any


SESSION_DIR = Path("memory/sessions")
PROFILE_DIR = Path("memory/profiles")

SESSION_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_CATEGORIES = [
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
]


def get_session_path(session_id: str) -> Path:
    """Return the JSON path for a conversation session."""
    return SESSION_DIR / f"{session_id}.json"


def get_profile_path(user_id: str) -> Path:
    """Return the JSON path for a user profile."""
    return PROFILE_DIR / f"{user_id}.json"


def load_session(session_id: str) -> list[dict[str, Any]]:
    """Load conversation history for one session."""
    path = get_session_path(session_id)

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_session(session_id: str, messages: list[dict[str, Any]]) -> None:
    """Persist conversation history for one session."""
    path = get_session_path(session_id)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(messages, file, indent=2)


def load_user_profile(user_id: str) -> dict[str, Any]:
    """Load distilled user profile facts."""
    path = get_profile_path(user_id)

    if not path.exists():
        return {
            "name": None,
            "frequent_topics": {},
            "preferences": [],
            "last_topics": [],
        }

    with open(path, "r", encoding="utf-8") as file:
        profile = json.load(file)

    profile.setdefault("name", None)
    profile.setdefault("frequent_topics", {})
    profile.setdefault("preferences", [])
    profile.setdefault("last_topics", [])
    return profile


def save_user_profile(user_id: str, profile: dict[str, Any]) -> None:
    """Persist distilled user profile facts."""
    path = get_profile_path(user_id)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)


def extract_topics(text: str) -> list[str]:
    """Find known dataset topics mentioned in text."""
    lowered = text.lower()
    topics: list[str] = []

    for topic in KNOWN_CATEGORIES:
        if topic in lowered or f"{topic}s" in lowered:
            category = topic.upper()
            if topic == "cancel":
                category = "CANCEL"
            topics.append(category)

    return topics


def update_user_profile_from_turn(user_id: str, question: str, answer: str) -> dict[str, Any]:
    """Update distilled profile facts from one completed turn."""
    profile = load_user_profile(user_id)
    topics = extract_topics(question)
    if not topics and "more" in question.lower():
        topics = extract_topics(answer)

    name_match = re.search(
        r"\bmy name is ([A-Za-z][A-Za-z -]{0,40}?)(?:\s+and\b|[.!?,]|$)",
        question,
        re.I,
    )
    if name_match:
        profile["name"] = name_match.group(1).strip()

    if "prefer" in question.lower():
        preferences = profile["preferences"]
        if question not in preferences:
            preferences.append(question)
        profile["preferences"] = preferences[-10:]

    frequent_topics = profile["frequent_topics"]
    for topic in topics:
        frequent_topics[topic] = int(frequent_topics.get(topic, 0)) + 1
    profile["frequent_topics"] = frequent_topics

    last_topics = profile["last_topics"]
    for topic in topics:
        if topic in last_topics:
            last_topics.remove(topic)
        last_topics.append(topic)
    profile["last_topics"] = last_topics[-5:]

    save_user_profile(user_id, profile)
    return profile


def get_last_user_topic(history: list[dict[str, Any]]) -> dict[str, str] | None:
    """Extract the last remembered dataset topic from conversation history."""
    for message in reversed(history):
        if message.get("role") != "user":
            continue

        topics = extract_topics(message.get("content", ""))
        if topics:
            return {
                "category": topics[-1],
                "question": message.get("content", ""),
            }

    return None


def get_last_two_answer_counts(history: list[dict[str, Any]]) -> list[int]:
    """Extract the most recent two integer counts from assistant answers."""
    counts: list[int] = []

    for message in reversed(history):
        if message.get("role") != "assistant":
            continue

        text = message.get("content", "")
        match = re.search(r"\bthere (?:are|is) ([0-9,]+)\b", text, re.I)
        if match:
            counts.append(int(match.group(1).replace(",", "")))

        if len(counts) == 2:
            break

    return counts
