"""Fast unit tests that need no LLM: tools run against a temporary database."""

import pytest

from src import tools
from src.agent import route_after_grade
from src.config import settings


# Point every test at a fresh temporary database so real data is never touched
@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "test.db")
    tools.init_db()


# An existing order is found even when the ID is typed in lowercase
def test_order_found():
    assert tools.get_order("ord-1001")["status"] == "Shipped"


# An unknown order ID returns None instead of raising an error
def test_order_missing():
    assert tools.get_order("ORD-9999") is None


# Creating a ticket returns an ID that starts with TKT-
def test_ticket_created():
    assert tools.create_ticket("help", "test").startswith("TKT-")


# When the grader says the context is relevant, the agent goes straight to answering
def test_relevant_goes_to_generate():
    assert route_after_grade({"relevant": True, "retries": 0}) == "generate"


# Weak retrieval triggers a rewrite first, then escalation once retries are used up
def test_weak_retrieval_rewrites_then_escalates():
    assert route_after_grade({"relevant": False, "retries": 0}) == "rewrite"
    assert (
        route_after_grade({"relevant": False, "retries": settings.MAX_RETRIES})
        == "escalate"
    )
