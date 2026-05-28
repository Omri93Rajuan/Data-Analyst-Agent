from typing import Any

from pydantic import BaseModel, Field

from src.data_loader import load_dataset


class CountRowsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Filter by category name."
    )

    intent: str | None = Field(
        default=None,
        description="Filter by intent name."
    )


def count_rows(
    category: str | None = None,
    intent: str | None = None,
) -> int:
    """
    Count rows in the dataset using optional filters.
    """
    df = load_dataset()

    if category:
        df = df[
            df["category"].str.lower() == category.lower()
        ]

    if intent:
        df = df[
            df["intent"].str.lower() == intent.lower()
        ]

    return len(df)


class ShowExamplesInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Filter by category."
    )

    intent: str | None = Field(
        default=None,
        description="Filter by intent."
    )

    limit: int = Field(
        default=3,
        description="Number of examples to return."
    )


def show_examples(
    category: str | None = None,
    intent: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """
    Return dataset examples.
    """
    df = load_dataset()

    if category:
        df = df[
            df["category"].str.lower() == category.lower()
        ]

    if intent:
        df = df[
            df["intent"].str.lower() == intent.lower()
        ]

    rows = df.head(limit)

    return rows[
        ["instruction", "category", "intent", "response"]
    ].to_dict(orient="records")