# Bitext Customer Service Data Analyst Agent

A Python CLI data analyst agent for the Bitext Customer Service dataset.

The main CLI path is intentionally deterministic: it routes the question with LangGraph, calls pandas-backed tools for factual answers, and prints a clear Thought / Action / Observation trace. This keeps counts and examples grounded in the CSV instead of relying on an LLM to guess. The project also includes an optional LangGraph prebuilt ReAct agent wired to the same tools and Nebius model for the assignment's ReAct requirement.

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

For each turn, the CLI prints:

- router decision
- Thought / Action / Observation reasoning steps
- selected handler/tool
- observations
- final answer

## Memory

Session chat history is stored separately from user profile facts:

```text
memory/checkpoints.sqlite
memory/sessions/<session_id>.json
memory/profiles/<user_id>.json
```

LangGraph state is persisted with `SqliteSaver` in `memory/checkpoints.sqlite`. The JSON files are kept as a simple, inspectable layer for conversation history and distilled user profile facts.

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

Example MCP client configuration:

```json
{
  "mcpServers": {
    "bitext-customer-service": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/agents"
    }
  }
}
```

After connecting a compatible MCP client, call a tool such as `get_categories` or `count_rows_tool`. Tool responses are structured data, not natural-language summaries.

## Architecture

```text
main.py
  -> src.graph.build_graph()
      -> router_node
      -> structured_node | unstructured_node | profile_node | out_of_scope_node
      -> profile_update_node
      -> LangGraph SqliteSaver checkpoint

src.react_agent
  -> optional LangGraph prebuilt ReAct agent path
  -> Nebius ChatOpenAI model
  -> same deterministic tools from TOOL_REGISTRY

src.query_handler
  -> deterministic handlers used by the default CLI path
  -> prints Thought / Action / Observation traces
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

The configured Nebius Token Factory model is `meta-llama/Llama-3.3-70B-Instruct` in `src/config.py`.

The default CLI does not use the LLM for factual dataset answers. That is a deliberate design choice: counts, distributions, and examples should come from deterministic pandas tools. The CLI still exposes the agent's reasoning path by printing Thought / Action / Observation steps for each supported query.

For a true LLM-driven ReAct path, `src.react_agent.build_react_agent()` builds a LangGraph prebuilt ReAct agent with the same Pydantic-described tools and the Nebius model. This keeps the ReAct implementation available without making the reliable CLI path depend on API availability.

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

- The default CLI uses deterministic tool selection, while the optional ReAct agent path is available in `src.react_agent`.
- Routing is rule-based and intentionally conservative in the default path.
- Only supported dataset question patterns are answered.
- Unsupported queries are declined clearly instead of guessed.
- Follow-up reasoning uses recent session history and known dataset topics, not semantic search.
- The profile is a simple distilled JSON profile, not a long-term vector memory.
