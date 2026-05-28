from typing import Any

from fastmcp import FastMCP

from src.tools import TOOL_REGISTRY


mcp = FastMCP("BitextCustomerServiceAgent")


@mcp.tool()
def get_categories() -> dict[str, list[str]]:
    """Return all dataset categories as structured data."""
    tool = TOOL_REGISTRY["list_categories"]
    categories = tool["function"]()
    return {"categories": categories}


@mcp.tool()
def count_rows_tool(category: str | None = None, intent: str | None = None) -> dict[str, Any]:
    """Count rows with optional category and intent filters."""
    tool = TOOL_REGISTRY["count_rows"]
    count = tool["function"](category=category, intent=intent)
    return {
        "category": category,
        "intent": intent,
        "count": count,
    }


@mcp.tool()
def show_examples_tool(
    category: str | None = None,
    intent: str | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """Return example rows with optional category and intent filters."""
    tool = TOOL_REGISTRY["show_examples"]
    examples = tool["function"](category=category, intent=intent, limit=limit)
    return {
        "category": category,
        "intent": intent,
        "limit": limit,
        "examples": examples,
    }


@mcp.tool()
def intent_distribution_tool(category: str) -> dict[str, Any]:
    """Return intent counts for a category."""
    tool = TOOL_REGISTRY["intent_distribution"]
    distribution = tool["function"](category=category)
    return {
        "category": category,
        "distribution": distribution,
    }


# Descriptions are kept in TOOL_REGISTRY so MCP wrappers stay consistent with the
# deterministic tool layer.


if __name__ == "__main__":
    mcp.run()
