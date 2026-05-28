from fastmcp import FastMCP

from src.tools import (
    count_rows,
    list_categories,
    show_examples,
)


mcp = FastMCP(
    "BitextCustomerServiceAgent"
)


@mcp.tool()
def get_categories() -> list[str]:
    """
    Return all dataset categories.
    """
    return list_categories()


@mcp.tool()
def get_refund_count() -> int:
    """
    Return number of refund rows.
    """
    return count_rows(
        category="REFUND"
    )


@mcp.tool()
def get_shipping_examples() -> list[dict]:
    """
    Return shipping examples.
    """
    return show_examples(
        category="SHIPPING",
        limit=3,
    )


if __name__ == "__main__":
    mcp.run()