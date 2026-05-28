# Bitext Customer Service Data Analyst Agent - Omri Rajuan

A Python CLI data analyst agent for the Bitext Customer Service dataset.

Submitted by: Omri Rajuan

Most of the questions in this assignment are about exact counts, examples, and distributions, so I kept the default path simple and data-first: route the question with LangGraph, call pandas tools, and print the steps the agent took. There is also a `--mode react` option that runs a LangGraph ReAct agent with the same tools and the Nebius model. If the API key is missing, it falls back to the deterministic path instead of crashing.

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
python main.py --mode react --session demo --user alice
```

Use `--mode deterministic` for the pandas-first path. Use `--mode react` for the LLM ReAct path. React mode needs `NEBIUS_API_KEY`; without it, the CLI prints the fallback reason and still answers through the deterministic graph.

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

For each turn, the CLI shows:

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

LangGraph state is saved with `SqliteSaver` in `memory/checkpoints.sqlite`. I also keep JSON files because they are easy to inspect while testing the session history and the user profile.

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
  -> --mode deterministic: src.graph.build_graph()
      -> router_node
      -> structured_node | unstructured_node | profile_node | out_of_scope_node
      -> profile_update_node
      -> LangGraph SqliteSaver checkpoint
  -> --mode react: src.react_agent.run_react_agent()
      -> LangGraph prebuilt ReAct agent
      -> Nebius ChatOpenAI model
      -> same dataset tools and schemas
      -> deterministic graph fallback

src.react_agent
  -> LangGraph prebuilt ReAct agent path
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

The default CLI does not use the LLM for factual dataset answers. I made that choice because counts, distributions, and examples should come directly from the CSV. The CLI still prints a Thought / Action / Observation trace so the tool path is visible.

For the LLM-driven ReAct path, run `python main.py --mode react`. This uses `src.react_agent.build_react_agent()`, a LangGraph prebuilt ReAct agent with the same tools and the Nebius model. If the Nebius key is unavailable, the CLI falls back to the deterministic graph so the project is still runnable.

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

Each public tool has a docstring, typed return value, pandas logic, and a matching Pydantic input schema in `src/tools.py`.

## Limitations

- The default CLI uses deterministic tool selection, while `--mode react` runs the LLM ReAct agent.
- Routing is rule-based and conservative in the default path.
- Only supported dataset question patterns are answered.
- Unsupported queries are declined clearly instead of guessed.
- Follow-up reasoning uses recent session history and known dataset topics, not semantic search.
- The profile is a simple distilled JSON profile, not a long-term vector memory.
