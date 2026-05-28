from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.memory import load_user_profile, update_user_profile_from_turn
from src.query_handler import analyze_structured_query, analyze_unstructured_query
from src.router import QueryType, route_query


class AgentState(TypedDict, total=False):
    """State passed between LangGraph workflow nodes."""

    question: str
    query_type: str
    answer: str
    session_id: str
    user_id: str
    history: list[dict[str, Any]]
    selected_tool: str
    observations: list[str]
    reasoning_steps: list[str]


def _append_step(state: AgentState, step: str) -> list[str]:
    steps = list(state.get("reasoning_steps", []))
    steps.append(step)
    return steps


def router_node(state: AgentState) -> AgentState:
    """Classify the question before choosing a handler."""
    query_type = route_query(state["question"]).value
    return {
        "query_type": query_type,
        "reasoning_steps": _append_step(state, f"router decision: {query_type}"),
    }


def structured_node(state: AgentState) -> AgentState:
    """Handle factual dataset questions with deterministic pandas tools."""
    result = analyze_structured_query(
        question=state["question"],
        history=state.get("history", []),
        user_id=state.get("user_id"),
    )
    return {
        "answer": result.answer,
        "selected_tool": result.selected_tool,
        "observations": result.observations,
        "reasoning_steps": _append_step(state, f"selected handler/tool: {result.selected_tool}"),
    }


def unstructured_node(state: AgentState) -> AgentState:
    """Handle supported summary-style dataset questions."""
    result = analyze_unstructured_query(state["question"])
    return {
        "answer": result.answer,
        "selected_tool": result.selected_tool,
        "observations": result.observations,
        "reasoning_steps": _append_step(state, f"selected handler/tool: {result.selected_tool}"),
    }


def out_of_scope_node(state: AgentState) -> AgentState:
    """Decline questions that are outside the dataset scope."""
    return {
        "answer": "I can only answer questions about the Bitext Customer Service dataset.",
        "selected_tool": "none",
        "observations": ["Question did not match supported dataset topics."],
        "reasoning_steps": _append_step(state, "selected handler/tool: out_of_scope_node"),
    }


def profile_node(state: AgentState) -> AgentState:
    """Handle user profile updates and memory recall."""
    question = state["question"]
    user_id = state.get("user_id")

    if "what do you remember about me" in question.lower():
        profile = load_user_profile(user_id or "default_user")
        frequent_topics = profile.get("frequent_topics", {})
        last_topics = profile.get("last_topics", [])
        preferences = profile.get("preferences", [])
        name = profile.get("name")

        lines = ["Here is what I remember about you:"]
        lines.append(f"- Name: {name}" if name else "- Name: not provided")
        lines.append(
            "- Recent dataset topics: " + (", ".join(last_topics) if last_topics else "none yet")
        )
        if frequent_topics:
            sorted_topics = sorted(frequent_topics.items(), key=lambda item: item[1], reverse=True)
            lines.append(
                "- Frequent topics: "
                + ", ".join(f"{topic} ({count})" for topic, count in sorted_topics[:5])
            )
        else:
            lines.append("- Frequent topics: none yet")
        lines.append(
            "- Preferences: " + ("; ".join(preferences[-3:]) if preferences else "none recorded")
        )
        answer = "\n".join(lines)
        selected_tool = "load_user_profile"
        observations = [f"Loaded profile for user '{user_id}'."]
    else:
        answer = "Got it — I updated your profile."
        selected_tool = "update_user_profile"
        observations = ["Profile-only statement accepted."]

    return {
        "answer": answer,
        "selected_tool": selected_tool,
        "observations": observations,
        "reasoning_steps": _append_step(state, f"selected handler/tool: {selected_tool}"),
    }


def profile_update_node(state: AgentState) -> AgentState:
    """Persist distilled profile facts after a completed turn."""
    user_id = state.get("user_id")
    if not user_id:
        return {
            "reasoning_steps": _append_step(state, "profile update skipped: no user_id"),
        }

    update_user_profile_from_turn(
        user_id=user_id,
        question=state["question"],
        answer=state.get("answer", ""),
    )
    return {
        "reasoning_steps": _append_step(state, f"profile updated for user: {user_id}"),
    }


def _route_after_router(state: AgentState) -> str:
    query_type = state.get("query_type")
    if query_type == QueryType.STRUCTURED.value:
        return "structured_node"
    if query_type == QueryType.UNSTRUCTURED.value:
        return "unstructured_node"
    if query_type == QueryType.PROFILE.value:
        return "profile_node"
    return "out_of_scope_node"


def build_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("router_node", router_node)
    workflow.add_node("structured_node", structured_node)
    workflow.add_node("unstructured_node", unstructured_node)
    workflow.add_node("out_of_scope_node", out_of_scope_node)
    workflow.add_node("profile_node", profile_node)
    workflow.add_node("profile_update_node", profile_update_node)

    workflow.set_entry_point("router_node")
    workflow.add_conditional_edges(
        "router_node",
        _route_after_router,
        {
            "structured_node": "structured_node",
            "unstructured_node": "unstructured_node",
            "profile_node": "profile_node",
            "out_of_scope_node": "out_of_scope_node",
        },
    )
    workflow.add_edge("structured_node", "profile_update_node")
    workflow.add_edge("unstructured_node", "profile_update_node")
    workflow.add_edge("out_of_scope_node", "profile_update_node")
    workflow.add_edge("profile_node", "profile_update_node")
    workflow.add_edge("profile_update_node", END)

    return workflow.compile()
