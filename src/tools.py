from collections.abc import Callable
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from src.data_loader import load_dataset


class ListCategoriesInput(BaseModel):
    """Input schema for list_categories."""


class ListIntentsInput(BaseModel):
    """Input schema for list_intents."""

    category: str | None = Field(
        default=None,
        description="Optional category filter, for example ACCOUNT or REFUND.",
    )


class CountRowsInput(BaseModel):
    """Input schema for count_rows."""

    category: str | None = Field(
        default=None,
        description="Optional exact category filter, for example REFUND, SHIPPING, ACCOUNT.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional exact intent filter, for example complaint, get_refund, edit_account. Do not invent broad intents like REQUEST.",
    )


class ShowExamplesInput(BaseModel):
    """Input schema for show_examples."""

    category: str | None = Field(default=None, description="Optional category filter.")
    intent: str | None = Field(default=None, description="Optional intent filter.")
    limit: int = Field(default=3, ge=1, le=20, description="Maximum rows to return.")
    offset: int = Field(default=0, ge=0, description="Rows to skip before returning examples.")


class IntentDistributionInput(BaseModel):
    """Input schema for intent_distribution."""

    category: str = Field(description="Category to summarize.")


class SearchInstructionsInput(BaseModel):
    """Input schema for search_instructions."""

    query: str = Field(description="Search text.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum rows to return.")


class TopIntentsInput(BaseModel):
    """Input schema for top_intents."""

    limit: int = Field(default=5, ge=1, le=20, description="Number of intents to return.")


class CategoryDistributionInput(BaseModel):
    """Input schema for category_distribution."""


class CompareCategoriesInput(BaseModel):
    """Input schema for compare_categories."""

    first_category: str = Field(description="First category to compare.")
    second_category: str = Field(description="Second category to compare.")


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


def list_categories() -> list[str]:
    """Return all category names in the Bitext customer service dataset."""
    df = load_dataset()
    return sorted(df["category"].dropna().unique().tolist())


def list_intents(category: str | None = None) -> list[str]:
    """Return all intent names, optionally filtered by category."""
    df = load_dataset()
    normalized_category = _normalize(category)

    if normalized_category:
        df = df[df["category"].str.lower() == normalized_category]

    return sorted(df["intent"].dropna().unique().tolist())


def count_rows(category: str | None = None, intent: str | None = None) -> int:
    """Count dataset rows, optionally filtered by category and/or intent."""
    df = load_dataset()
    normalized_category = _normalize(category)
    normalized_intent = _normalize(intent)

    if normalized_category:
        df = df[df["category"].str.lower() == normalized_category]

    if normalized_intent:
        df = df[df["intent"].str.lower() == normalized_intent]

    return int(len(df))


def show_examples(
    category: str | None = None,
    intent: str | None = None,
    limit: int = 3,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return example customer instructions and responses from the dataset."""
    df = load_dataset()
    normalized_category = _normalize(category)
    normalized_intent = _normalize(intent)

    if normalized_category:
        df = df[df["category"].str.lower() == normalized_category]

    if normalized_intent:
        df = df[df["intent"].str.lower() == normalized_intent]

    safe_limit = max(1, min(limit, 20))
    safe_offset = max(0, offset)

    return (
        df[["instruction", "category", "intent", "response"]]
        .iloc[safe_offset : safe_offset + safe_limit]
        .to_dict(orient="records")
    )


def intent_distribution(category: str) -> dict[str, int]:
    """Return intent counts for one category."""
    df = load_dataset()
    normalized_category = _normalize(category)
    filtered = df[df["category"].str.lower() == normalized_category]
    return {str(intent): int(count) for intent, count in filtered["intent"].value_counts().items()}


def search_instructions(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search instructions, intents, and categories for matching rows."""
    df = load_dataset()
    q = query.strip().lower()

    if not q:
        return []

    filtered = df[
        df["instruction"].str.lower().str.contains(q, na=False, regex=False)
        | df["intent"].str.lower().str.contains(q, na=False, regex=False)
        | df["category"].str.lower().str.contains(q, na=False, regex=False)
    ]

    safe_limit = max(1, min(limit, 20))
    return (
        filtered[["instruction", "category", "intent", "response"]]
        .head(safe_limit)
        .to_dict(orient="records")
    )


def top_intents(limit: int = 5) -> dict[str, int]:
    """Return the most common intents in the dataset."""
    df = load_dataset()
    safe_limit = max(1, min(limit, 20))
    return {str(intent): int(count) for intent, count in df["intent"].value_counts().head(safe_limit).items()}


def category_distribution() -> dict[str, int]:
    """Return row counts by category."""
    df = load_dataset()
    return {
        str(category): int(count)
        for category, count in df["category"].value_counts().sort_index().items()
    }


def compare_categories(first_category: str, second_category: str) -> dict[str, Any]:
    """Compare two categories by row count and top intents."""
    first = first_category.strip().upper()
    second = second_category.strip().upper()

    return {
        first: {
            "row_count": count_rows(category=first),
            "intent_distribution": intent_distribution(first),
        },
        second: {
            "row_count": count_rows(category=second),
            "intent_distribution": intent_distribution(second),
        },
    }


class ToolRegistryEntry(TypedDict):
    """Registry metadata for a deterministic dataset tool."""

    function: Callable[..., Any]
    schema: type[BaseModel]
    description: str


TOOL_REGISTRY: dict[str, ToolRegistryEntry] = {
    "list_categories": {
        "function": list_categories,
        "schema": ListCategoriesInput,
        "description": "Return all category names in the Bitext customer service dataset.",
    },
    "list_intents": {
        "function": list_intents,
        "schema": ListIntentsInput,
        "description": "Return all intent names, optionally filtered by category.",
    },
    "count_rows": {
        "function": count_rows,
        "schema": CountRowsInput,
        "description": "Count dataset rows with optional exact category and/or exact intent filters. Use category=REFUND for broad questions like refund requests.",
    },
    "show_examples": {
        "function": show_examples,
        "schema": ShowExamplesInput,
        "description": "Return example customer instructions and responses from the dataset.",
    },
    "intent_distribution": {
        "function": intent_distribution,
        "schema": IntentDistributionInput,
        "description": "Return intent counts for one category.",
    },
    "search_instructions": {
        "function": search_instructions,
        "schema": SearchInstructionsInput,
        "description": "Search instructions, intents, and categories for matching rows.",
    },
    "top_intents": {
        "function": top_intents,
        "schema": TopIntentsInput,
        "description": "Return the most common intents in the dataset.",
    },
    "category_distribution": {
        "function": category_distribution,
        "schema": CategoryDistributionInput,
        "description": "Return row counts by category.",
    },
    "compare_categories": {
        "function": compare_categories,
        "schema": CompareCategoriesInput,
        "description": "Compare two categories by row count and top intents.",
    },
}
