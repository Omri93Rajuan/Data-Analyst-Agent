from src.memory import load_session, save_session
from src.query_handler import handle_structured_query
from src.router import QueryType, route_query
from src.tools import count_rows, intent_distribution, list_categories


def test_router_out_of_scope() -> None:
    assert route_query("What is the capital of France?") == QueryType.OUT_OF_SCOPE


def test_router_structured_refund_query() -> None:
    assert route_query("How many refund requests did we get?") == QueryType.STRUCTURED


def test_router_profile_statement() -> None:
    assert route_query("My name is Alex and I prefer concise answers") == QueryType.PROFILE


def test_list_categories_includes_refund() -> None:
    assert "REFUND" in list_categories()


def test_count_rows_refund() -> None:
    assert count_rows(category="REFUND") == 2992


def test_account_intent_distribution_contains_edit_account() -> None:
    distribution = intent_distribution("ACCOUNT")
    assert distribution["edit_account"] == 1000


def test_handle_structured_refund_count() -> None:
    answer = handle_structured_query("How many refund requests did we get?", [])
    assert "2992" in answer


def test_memory_save_load_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.memory.SESSION_DIR", tmp_path)
    messages = [{"role": "user", "content": "hello"}]

    save_session("pytest-session", messages)

    assert load_session("pytest-session") == messages
