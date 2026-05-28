from src.tools import (
    count_rows,
    intent_distribution,
    list_categories,
    list_intents,
    search_instructions,
    show_examples,
)


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return "No matching examples found."

    return "\n\n".join(
        f"- Customer: {example['instruction']}\n"
        f"  Category: {example['category']}\n"
        f"  Intent: {example['intent']}"
        for example in examples
    )


def _extract_limit(question: str, default: int = 3) -> int:
    for word in question.split():
        if word.isdigit():
            return int(word)

    return default


def handle_structured_query(question: str) -> str:
    q = question.lower()
    limit = _extract_limit(question)

    if "what categories" in q or "categories exist" in q or "list categories" in q:
        categories = list_categories()
        return "Categories in the dataset:\n\n" + "\n".join(f"- {c}" for c in categories)

    if "intent" in q and "account" in q and "distribution" in q:
        distribution = intent_distribution("ACCOUNT")
        return "Intent distribution in ACCOUNT:\n\n" + "\n".join(
            f"- {intent}: {count}" for intent, count in distribution.items()
        )

    if "intent" in q and "account" in q:
        intents = list_intents("ACCOUNT")
        return "ACCOUNT intents:\n\n" + "\n".join(f"- {intent}" for intent in intents)

    if "refund" in q and ("how many" in q or "count" in q):
        count = count_rows(category="REFUND")
        return f"There are {count} refund-related rows in the dataset."

    if "complaint" in q and ("how many" in q or "count" in q):
        count = count_rows(intent="complaint")
        return f"There are {count} complaint rows in the dataset."

    if "shipping" in q and "example" in q:
        examples = show_examples(category="SHIPPING", limit=limit)
        return f"Here are {limit} SHIPPING examples:\n\n{_format_examples(examples)}"

    if "refund" in q and "example" in q:
        examples = show_examples(category="REFUND", limit=limit)
        return f"Here are {limit} REFUND examples:\n\n{_format_examples(examples)}"

    if "money back" in q or "wanting their money back" in q:
        examples = search_instructions("refund", limit=limit)
        return f"Examples of customers wanting their money back:\n\n{_format_examples(examples)}"

    if "distribution" in q and "account" in q:
        distribution = intent_distribution("ACCOUNT")
        return "Intent distribution in ACCOUNT:\n\n" + "\n".join(
            f"- {intent}: {count}" for intent, count in distribution.items()
        )

    return (
        "I understand this is a structured dataset question, "
        "but this exact query is not supported yet."
    )


def handle_unstructured_query(question: str) -> str:
    q = question.lower()

    if "feedback" in q:
        examples = show_examples(category="FEEDBACK", limit=10)
        return (
            "The FEEDBACK category mainly contains customer reviews and complaints. "
            "Customers either provide negative feedback, submit complaints, or leave reviews. "
            "Typical agent responses acknowledge the feedback, show empathy, and ask for more "
            "details so the issue can be handled.\n\n"
            "Sample rows:\n\n"
            f"{_format_examples(examples[:3])}"
        )

    if "complaint" in q:
        examples = show_examples(intent="complaint", limit=10)
        return (
            "Complaint intents usually involve customers expressing dissatisfaction. "
            "Agent responses tend to be empathetic, apologize for the experience, "
            "and invite the customer to share details so support can resolve the issue.\n\n"
            "Sample rows:\n\n"
            f"{_format_examples(examples[:3])}"
        )

    if "cancellation" in q or "cancel" in q:
        examples = search_instructions("cancel", limit=10)
        return (
            "Cancellation-related responses usually acknowledge that the customer wants "
            "to cancel an order, purchase, or subscription. The representative typically "
            "confirms the request, offers help, and asks for the relevant order or account details.\n\n"
            "Sample rows:\n\n"
            f"{_format_examples(examples[:3])}"
        )

    return (
        "This open-ended dataset question is not supported yet. "
        "Try asking about FEEDBACK, complaints, refunds, shipping, or cancellations."
    )