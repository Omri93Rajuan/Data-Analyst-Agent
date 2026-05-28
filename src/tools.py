from typing import Any

from src.data_loader import load_dataset


def list_categories() -> list[str]:
    df = load_dataset()
    return sorted(df["category"].dropna().unique().tolist())


def list_intents(category: str | None = None) -> list[str]:
    df = load_dataset()

    if category:
        df = df[df["category"].str.lower() == category.lower()]

    return sorted(df["intent"].dropna().unique().tolist())


def count_rows(
    category: str | None = None,
    intent: str | None = None,
) -> int:
    df = load_dataset()

    if category:
        df = df[df["category"].str.lower() == category.lower()]

    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]

    return len(df)


def show_examples(
    category: str | None = None,
    intent: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    df = load_dataset()

    if category:
        df = df[df["category"].str.lower() == category.lower()]

    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]

    return df[["instruction", "category", "intent", "response"]].head(limit).to_dict(
        orient="records"
    )


def intent_distribution(category: str) -> dict[str, int]:
    df = load_dataset()
    filtered = df[df["category"].str.lower() == category.lower()]
    return filtered["intent"].value_counts().to_dict()


def search_instructions(query: str, limit: int = 5) -> list[dict[str, Any]]:
    df = load_dataset()
    q = query.lower()

    filtered = df[
        df["instruction"].str.lower().str.contains(q, na=False)
        | df["intent"].str.lower().str.contains(q, na=False)
        | df["category"].str.lower().str.contains(q, na=False)
    ]

    return filtered[["instruction", "category", "intent", "response"]].head(limit).to_dict(
        orient="records"
    )