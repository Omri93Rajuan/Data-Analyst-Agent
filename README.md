# Bitext Customer Service Data Analyst Agent

Python CLI data analyst agent for the Bitext Customer Service dataset. It uses deterministic pandas tools for factual answers, a LangGraph workflow for routing, JSON memory for sessions and user profiles, and a FastMCP server for external tool access.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Placement

Place the CSV at:

```text
data/bitext_customer_service.csv
```

Expected columns:

```text
flags, instruction, category, intent, response
```

## CLI Usage

```bash
python main.py
python main.py --session demo
python main.py --session demo --user alice
```

Example questions:

```text
What categories exist?
Show me 3 examples from REFUND
Show me 3 more
How many complaints did we get?
What about refunds?
What is the total count of the last two?
Summarize complaint responses.
What do you remember about me?
```

The CLI prints:

- router decision
- selected handler/tool
- observations
- final answer

## Memory

Session chat history is stored separately from user profile facts.

```text
memory/sessions/<session_id>.json
memory/profiles/<user_id>.json
```

The profile stores distilled facts only:

- name
- frequent_topics
- preferences
- last_topics

Try:

```bash
python main.py --session demo --user alex
```

Then ask:

```text
My name is Alex and I prefer concise answers.
What do you remember about me?
```

Restart the CLI with the same `--session` and `--user` to verify persistence.

## MCP Server

Run:

```bash
python -m src.mcp_server
```

Exposed MCP tools return structured data:

- `get_categories`
- `count_rows_tool`
- `show_examples_tool`
- `intent_distribution_tool`

## Architecture

```text
main.py
  -> src.graph.build_graph()
      -> router_node
      -> structured_node | unstructured_node | out_of_scope_node
      -> profile_update_node

src.query_handler
  -> deterministic pattern handlers
  -> calls src.tools

src.tools
  -> pandas operations over data/bitext_customer_service.csv
  -> Pydantic input schemas for public tools

src.memory
  -> JSON session history
  -> JSON distilled user profiles

src.mcp_server
  -> FastMCP wrapper around dataset tools
```

## Model Choice

The final workflow does not require an LLM for factual dataset answers. This is intentional: assignment questions are best answered by deterministic pandas tools to avoid hallucinated counts or unsupported claims. The project keeps `langchain` and `langgraph` dependencies because LangGraph provides the workflow orchestration, routing, and recursion controls.

## Tools List

- `list_categories`
- `list_intents`
- `count_rows`
- `show_examples`
- `intent_distribution`
- `search_instructions`
- `top_intents`
- `category_distribution`
- `compare_categories`

Each public tool has a docstring, typed return value, deterministic pandas logic, and a matching Pydantic input schema in `src/tools.py`.

## Limitations

- Routing is rule-based and intentionally conservative.
- Only supported dataset question patterns are answered.
- Unsupported queries are declined clearly instead of guessed.
- Follow-up reasoning uses recent session history and known dataset topics, not semantic search.
- The profile is a simple distilled JSON profile, not a long-term vector memory.
