import re
from dataclasses import dataclass
from typing import Any

from src.memory import get_last_two_answer_counts, get_last_user_topic, load_user_profile
from src.tools import (
    category_distribution,
    compare_categories,
    count_rows,
    intent_distribution,
    list_categories,
    list_intents,
    search_instructions,
    show_examples,
    top_intents,
)


@dataclass
class QueryResult:
    """Deterministic answer plus reasoning details for the CLI."""

    answer: str
    selected_tool: str
    observations: list[str]


def _format_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "No matching examples found."

    return "\n\n".join(
        f"- Customer: {example['instruction']}\n"
        f"  Category: {example['category']}\n"
        f"  Intent: {example['intent']}"
        for example in examples
    )


def _extract_limit(question: str, default: int = 3) -> int:
    for word in re.findall(r"\d+", question):
        return max(1, min(int(word), 20))

    return default


def _extract_category(question: str, history: list[dict[str, Any]] | None = None) -> str | None:
    q = question.lower()
    aliases = {
        "refund": "REFUND",
        "refunds": "REFUND",
        "shipping": "SHIPPING",
        "account": "ACCOUNT",
        "accounts": "ACCOUNT",
        "order": "ORDER",
        "orders": "ORDER",
        "invoice": "INVOICE",
        "invoices": "INVOICE",
        "payment": "PAYMENT",
        "payments": "PAYMENT",
        "feedback": "FEEDBACK",
        "delivery": "DELIVERY",
        "subscription": "SUBSCRIPTION",
        "subscriptions": "SUBSCRIPTION",
        "cancel": "CANCEL",
        "cancellation": "CANCEL",
        "contact": "CONTACT",
    }

    for word, category in aliases.items():
        if word in q:
            return category

    if "what about" in q and history:
        last_topic = get_last_user_topic(history)
        if last_topic:
            return last_topic["category"]

    return None


def _last_example_offset(history: list[dict[str, Any]], category: str | None) -> int:
    if not category:
        return 0

    offset = 0
    for message in reversed(history):
        content = message.get("content", "")
        if message.get("role") == "assistant" and category in content:
            offset += content.count("- Customer:")
        if message.get("role") == "user" and "example" in content.lower() and category.lower() in content.lower():
            break

    return offset


def _profile_answer(user_id: str | None) -> QueryResult:
    if not user_id:
        return QueryResult(
            answer="I do not have a user ID for this session, so I cannot load a profile.",
            selected_tool="load_user_profile",
            observations=["Missing user_id."],
        )

    profile = load_user_profile(user_id)
    frequent_topics = profile.get("frequent_topics", {})
    last_topics = profile.get("last_topics", [])
    preferences = profile.get("preferences", [])
    name = profile.get("name")

    lines = ["Here is what I remember about you:"]
    lines.append(f"- Name: {name}" if name else "- Name: not provided")
    lines.append(
        "- Recent dataset topics: " + (", ".join(last_topics) if last_topics else "none yet")
    )
    if frequent_topics:
        sorted_topics = sorted(frequent_topics.items(), key=lambda item: item[1], reverse=True)
        lines.append(
            "- Frequent topics: "
            + ", ".join(f"{topic} ({count})" for topic, count in sorted_topics[:5])
        )
    else:
        lines.append("- Frequent topics: none yet")
    lines.append(
        "- Preferences: " + ("; ".join(preferences[-3:]) if preferences else "none recorded")
    )

    return QueryResult(
        answer="\n".join(lines),
        selected_tool="load_user_profile",
        observations=[f"Loaded profile for user '{user_id}'."],
    )


def analyze_structured_query(
    question: str,
    history: list[dict[str, Any]] | None = None,
    user_id: str | None = None,
) -> QueryResult:
    """Answer supported factual dataset questions with deterministic tools."""
    history = history or []
    q = question.lower()
    limit = _extract_limit(question)
    category = _extract_category(question, history)

    if "remember about me" in q:
        return _profile_answer(user_id)

    if "total count of the last two" in q:
        counts = get_last_two_answer_counts(history)
        if len(counts) < 2:
            return QueryResult(
                answer="I do not have two previous count answers to add yet.",
                selected_tool="session_history",
                observations=["Fewer than two previous count answers found."],
            )
        total = sum(counts)
        return QueryResult(
            answer=f"The total count of the last two count answers is {total}.",
            selected_tool="session_history",
            observations=[f"Added counts {counts[0]} and {counts[1]}."],
        )

    if "more" in q and history:
        last_topic = get_last_user_topic(history)
        if last_topic:
            category = last_topic["category"]
            offset = _last_example_offset(history, category)
            examples = show_examples(category=category, limit=limit, offset=offset)
            return QueryResult(
                answer=f"Here are {limit} more examples from {category}:\n\n{_format_examples(examples)}",
                selected_tool="show_examples",
                observations=[f"category={category}", f"limit={limit}", f"offset={offset}"],
            )

    if "what categories" in q or "categories exist" in q or "list categories" in q:
        categories = list_categories()
        return QueryResult(
            answer="Categories in the dataset:\n\n" + "\n".join(f"- {c}" for c in categories),
            selected_tool="list_categories",
            observations=[f"Found {len(categories)} categories."],
        )

    if "category distribution" in q or ("distribution" in q and "categor" in q):
        distribution = category_distribution()
        return QueryResult(
            answer="Category distribution:\n\n"
            + "\n".join(f"- {name}: {count}" for name, count in distribution.items()),
            selected_tool="category_distribution",
            observations=[f"Summarized {len(distribution)} categories."],
        )

    if "top intent" in q or "most common intent" in q:
        limit = _extract_limit(question, default=5)
        distribution = top_intents(limit=limit)
        return QueryResult(
            answer=f"Top {limit} intents:\n\n"
            + "\n".join(f"- {intent}: {count}" for intent, count in distribution.items()),
            selected_tool="top_intents",
            observations=[f"limit={limit}"],
        )

    if "compare" in q and category:
        mentioned = [item for item in list_categories() if item.lower() in q]
        if len(mentioned) >= 2:
            comparison = compare_categories(mentioned[0], mentioned[1])
            lines = []
            for name, details in comparison.items():
                top_three = list(details["intent_distribution"].items())[:3]
                top_text = ", ".join(f"{intent}: {count}" for intent, count in top_three)
                lines.append(f"- {name}: {details['row_count']} rows; top intents: {top_text}")
            return QueryResult(
                answer="Category comparison:\n\n" + "\n".join(lines),
                selected_tool="compare_categories",
                observations=[f"categories={mentioned[0]}, {mentioned[1]}"],
            )

    if "intent" in q and "distribution" in q and category:
        distribution = intent_distribution(category)
        return QueryResult(
            answer=f"Intent distribution in {category}:\n\n"
            + "\n".join(f"- {intent}: {count}" for intent, count in distribution.items()),
            selected_tool="intent_distribution",
            observations=[f"category={category}", f"Found {len(distribution)} intents."],
        )

    if "intent" in q and category:
        intents = list_intents(category)
        return QueryResult(
            answer=f"{category} intents:\n\n" + "\n".join(f"- {intent}" for intent in intents),
            selected_tool="list_intents",
            observations=[f"category={category}", f"Found {len(intents)} intents."],
        )

    if "complaint" in q and ("how many" in q or "count" in q):
        row_count = count_rows(intent="complaint")
        return QueryResult(
            answer=f"There are {row_count} complaint rows in the dataset.",
            selected_tool="count_rows",
            observations=["intent=complaint"],
        )

    if category and ("how many" in q or "count" in q or q.strip().startswith("what about")):
        row_count = count_rows(category=category)
        return QueryResult(
            answer=f"There are {row_count} {category} rows in the dataset.",
            selected_tool="count_rows",
            observations=[f"category={category}"],
        )

    if category and "example" in q:
        examples = show_examples(category=category, limit=limit)
        return QueryResult(
            answer=f"Here are {limit} {category} examples:\n\n{_format_examples(examples)}",
            selected_tool="show_examples",
            observations=[f"category={category}", f"limit={limit}"],
        )

    if "money back" in q or "wanting their money back" in q:
        examples = search_instructions("refund", limit=limit)
        return QueryResult(
            answer=f"Examples of customers wanting their money back:\n\n{_format_examples(examples)}",
            selected_tool="search_instructions",
            observations=[f"query=refund", f"limit={limit}"],
        )

    return QueryResult(
        answer=(
            "I understand this is a structured dataset question, "
            "but this exact query is not supported yet."
        ),
        selected_tool="none",
        observations=["No supported deterministic pattern matched."],
    )


def analyze_unstructured_query(question: str) -> QueryResult:
    """Answer supported summary-style dataset questions with deterministic examples."""
    q = question.lower()

    if "feedback" in q:
        examples = show_examples(category="FEEDBACK", limit=10)
        return QueryResult(
            answer=(
                "The FEEDBACK category mainly contains customer reviews and complaints. "
                "Agent responses acknowledge the feedback, show empathy, and ask for more details.\n\n"
                "Sample rows:\n\n"
                f"{_format_examples(examples[:3])}"
            ),
            selected_tool="show_examples",
            observations=["category=FEEDBACK", "limit=10"],
        )

    if "complaint" in q:
        examples = show_examples(intent="complaint", limit=10)
        return QueryResult(
            answer=(
                "Complaint intents usually involve customers expressing dissatisfaction. "
                "Agent responses tend to apologize, show empathy, and ask for details needed to resolve the issue.\n\n"
                "Sample rows:\n\n"
                f"{_format_examples(examples[:3])}"
            ),
            selected_tool="show_examples",
            observations=["intent=complaint", "limit=10"],
        )

    if "cancellation" in q or "cancel" in q:
        examples = search_instructions("cancel", limit=10)
        return QueryResult(
            answer=(
                "Cancellation-related responses usually confirm the customer wants to cancel an order, purchase, "
                "or subscription. The representative offers help and asks for relevant order or account details.\n\n"
                "Sample rows:\n\n"
                f"{_format_examples(examples[:3])}"
            ),
            selected_tool="search_instructions",
            observations=["query=cancel", "limit=10"],
        )

    return QueryResult(
        answer=(
            "This open-ended dataset question is not supported yet. "
            "Try asking about FEEDBACK, complaints, refunds, shipping, or cancellations."
        ),
        selected_tool="none",
        observations=["No supported summary pattern matched."],
    )


def handle_structured_query(question: str, history: list[dict[str, Any]] | None = None) -> str:
    """Backward-compatible wrapper for structured questions."""
    return analyze_structured_query(question, history).answer


def handle_unstructured_query(question: str) -> str:
    """Backward-compatible wrapper for unstructured questions."""
    return analyze_unstructured_query(question).answer
