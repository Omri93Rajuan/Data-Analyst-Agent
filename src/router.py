from enum import Enum


class QueryType(str, Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    PROFILE = "profile"
    OUT_OF_SCOPE = "out_of_scope"


def route_query(question: str) -> QueryType:
    """
    Classify a user question before the agent chooses tools.
    This is a simple rule-based router for the first version.
    """
    q = question.lower()

    profile_words = [
        "my name is",
        "i prefer",
        "remember that",
        "what do you remember about me",
    ]

    if any(word in q for word in profile_words):
        return QueryType.PROFILE

    dataset_words = [
        "category", "categories", "intent", "intents",
        "refund", "shipping", "account", "order",
        "invoice", "payment", "feedback", "delivery",
        "subscription", "cancel", "complaint",
        "examples", "dataset", "customer", "response",
        "more", "remember about me", "what about",
        "total count", "last two",
    ]

    structured_words = [
        "how many", "count", "distribution", "list",
        "show me", "examples", "what categories",
        "what intents", "what about", "total count",
    ]

    unstructured_words = [
        "summarize", "summary", "typically respond",
        "how do agents respond", "explain",
    ]

    if not any(word in q for word in dataset_words):
        return QueryType.OUT_OF_SCOPE

    if any(word in q for word in unstructured_words):
        return QueryType.UNSTRUCTURED

    if any(word in q for word in structured_words):
        return QueryType.STRUCTURED

    return QueryType.STRUCTURED
