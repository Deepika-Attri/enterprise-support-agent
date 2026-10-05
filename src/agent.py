"""LangGraph agent: router -> knowledge (corrective RAG) | account (tool) | escalate."""

import argparse
import json
import re
from functools import lru_cache
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from src.config import get_llm, settings
from src.ingestion import ensure_index
from src.prompts import ACCOUNT_PROMPT, ANSWER_PROMPT, REWRITER_PROMPT, ROUTER_PROMPT
from src.retrieval import format_context, format_sources, grade_documents, retrieve
from src.tools import create_ticket, get_order, init_db


# ---------- State: the shared notebook every node reads and writes ----------
class AgentState(TypedDict, total=False):
    """Data passed between nodes; each node returns only the fields it changes."""

    question: str
    department: str | None
    include_samples: bool  # False = search uploaded documents only
    route: str  # knowledge | account | escalate | out_of_scope
    search_query: str  # current (maybe rewritten) query
    documents: list[Document]
    relevant: bool  # grader verdict
    retries: int  # rewrites used so far
    answer: str
    sources: list[str]
    ticket_id: str


def ask(prompt, **inputs) -> str:
    """Run prompt -> LLM -> clean string."""
    return (prompt | get_llm() | StrOutputParser()).invoke(inputs).strip()


def _rule_based_route(question: str) -> str | None:
    """Handle high-confidence support intents before asking a fallible classifier.

    This is deliberately small: the model still handles unusual wording, while the
    routes that must not fail (an order number, a request for a person, and policy
    topics) do not depend on model formatting or quality.
    """
    text = question.lower()
    if re.search(r"\bord-\d+\b", text):
        return "account"
    if re.search(
        r"\b(human|agent|representative|manager|supervisor|legal|lawyer|sue|lawsuit)\b",
        text,
    ):
        return "escalate"
    if re.search(
        r"\b(refund|return|exchange|shipping|delivery|damaged|replacement|"
        r"leave|holiday|vacation|benefit|policy|discount)\w*\b",
        text,
    ):
        return "knowledge"
    return None


# ---------- Nodes: each returns only the fields it changes ----------
def router_node(state: AgentState) -> dict:
    """Classify the question into knowledge / account / escalate / out_of_scope."""
    rule_route = _rule_based_route(state["question"])
    if rule_route:
        return {"route": rule_route}

    try:
        raw = ask(ROUTER_PROMPT, question=state["question"]).lower()
    except Exception:
        # A support question is the least surprising safe fallback; retrieval can
        # still provide a grounded answer or create a ticket when it cannot.
        return {"route": "knowledge"}
    valid = ["account", "escalate", "out_of_scope", "knowledge"]
    return {"route": next((r for r in valid if r in raw), "knowledge")}  # safe fallback


def retrieve_node(state: AgentState) -> dict:
    """Search ChromaDB with the current query, honouring department and sample filters."""
    docs = retrieve(
        state["search_query"],
        state.get("department"),
        include_samples=state.get("include_samples", True),
    )
    return {"documents": docs}


def grade_node(state: AgentState) -> dict:
    """Ask the grader whether the retrieved chunks can answer the question."""
    return {"relevant": grade_documents(state["question"], state["documents"])}


def rewrite_node(state: AgentState) -> dict:
    """Corrective RAG: rewrite the query and count the retry."""
    new_query = ask(
        REWRITER_PROMPT, question=state["question"], query=state["search_query"]
    )
    return {"search_query": new_query, "retries": state.get("retries", 0) + 1}


def extractive_answer(question: str, docs: list[Document]) -> str:
def fallback_answer(docs: list[Document]) -> str:
    """Return a readable, grounded fallback when answer generation is unavailable."""
    question_words = {
        word.rstrip("s")
        for word in re.findall(r"[a-z]{3,}", question.lower())
        if word
        not in {
            "about",
            "after",
            "does",
            "from",
            "have",
            "how",
            "long",
            "that",
            "the",
            "this",
            "what",
            "when",
            "with",
            "would",
            "your",
        }
    }
    candidates: list[tuple[int, str, str]] = []
    for doc in docs:
        source = doc.metadata.get("source", "available documents")
        for sentence in re.split(r"(?<=[.!?])\s+", doc.page_content.strip()):
            words = {
                word.rstrip("s")
                for word in re.findall(r"[a-z]{3,}", sentence.lower())
            }
            candidates.append((len(question_words & words), sentence.strip(), source))

    if not candidates:
    if not docs:
        return "I couldn't find an answer in the available documents."
    _, sentence, source = max(candidates, key=lambda item: item[0])
    return f"Based on the available policy: {sentence} [{source}]"
    doc = docs[0]
    source = doc.metadata.get("source", "available documents")
    return f"Based on the available policy: {doc.page_content.strip()} [{source}]"


def generate_node(state: AgentState) -> dict:
    """Write the final cited answer from the approved chunks."""
    docs = state["documents"]
    try:
        answer = ask(
            ANSWER_PROMPT, context=format_context(docs), question=state["question"]
        )
    except Exception:
        # A retrieved policy is more useful than an error page if the configured
        # model is temporarily unavailable. The source list still makes this
        # response auditable.
        answer = ""
    if not answer or answer.lower().startswith("i don't know based on"):
        answer = extractive_answer(state["question"], docs)
        answer = fallback_answer(docs)
    return {"answer": answer, "sources": format_sources(docs)}


def account_node(state: AgentState) -> dict:
    """Tool use: a regex finds the order ID (more reliable than a small LLM picking tool args)."""
    match = re.search(r"ORD-\d+", state["question"], re.IGNORECASE)
    if not match:
        return {
            "answer": "Please share your order ID (for example ORD-1001).",
            "sources": [],
        }

    order = get_order(match.group())
    if not order:
        return {
            "answer": f"I couldn't find order {match.group().upper()}.",
            "sources": [],
        }

    answer = ask(ACCOUNT_PROMPT, order=json.dumps(order), question=state["question"])
    return {"answer": answer, "sources": ["orders_db"]}


def escalate_node(state: AgentState) -> dict:
    """Human handoff: creates a real ticket in SQLite."""
    if state.get("route") == "escalate":
        reason = "Customer asked for a human or raised a complaint"
    else:
        reason = (
            f"No relevant documents after {state.get('retries', 0)} query rewrite(s)"
        )
    ticket_id = create_ticket(state["question"], reason)
    return {
        "answer": f"I've created ticket {ticket_id}. A human agent will follow up with you.",
        "ticket_id": ticket_id,
        "sources": [],
    }


def out_of_scope_node(state: AgentState) -> dict:
    """Politely refuse questions unrelated to the company."""
    return {
        "answer": "Sorry, I can only help with company policies, orders and support questions.",
        "sources": [],
    }


def no_answer_node(state: AgentState) -> dict:
    """End a failed knowledge search without opening an unnecessary ticket."""
    return {
        "answer": (
            "I couldn't find an answer to that in the available documents. "
            "Please try rephrasing your question or add a relevant policy document."
        ),
        "sources": [],
    }


# ---------- Decisions: conditional edges ----------
def route_after_grade(state: AgentState) -> str:
    """Good context -> answer. Weak -> rewrite. Retries used up -> escalate."""
    """Good context -> answer. Weak -> rewrite. Exhausted search -> no-answer."""
    if state["relevant"]:
        return "generate"
    return "rewrite" if state.get("retries", 0) < settings.MAX_RETRIES else "escalate"
    return "rewrite" if state.get("retries", 0) < settings.MAX_RETRIES else "no_answer"


# ---------- Graph assembly ----------
@lru_cache(maxsize=1)
def build_graph():
    """Wire the nodes and decisions into a compiled LangGraph workflow (built once)."""
    g = StateGraph(AgentState)

    for name, fn in {
        "router": router_node,
        "retrieve": retrieve_node,
        "grade": grade_node,
        "rewrite": rewrite_node,
        "generate": generate_node,
        "account": account_node,
        "escalate": escalate_node,
        "out_of_scope": out_of_scope_node,
        "no_answer": no_answer_node,
    }.items():
        g.add_node(name, fn)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "knowledge": "retrieve",
            "account": "account",
            "escalate": "escalate",
            "out_of_scope": "out_of_scope",
        },
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "escalate": "escalate",
            "no_answer": "no_answer",
        },
    )
    g.add_edge("rewrite", "retrieve")  # the self-correction loop

    for terminal in ("generate", "account", "escalate", "out_of_scope"):
    for terminal in ("generate", "account", "escalate", "out_of_scope", "no_answer"):
        g.add_edge(terminal, END)

    return g.compile()


# ---------- Public functions (used by the CLI, Streamlit and RAGAS later) ----------
def run_agent(
    question: str, department: str | None = None, include_samples: bool = True
) -> dict:
    """Run the agent; returns the answer plus contexts and the node path (RAGAS-ready)."""
    init_db()
    ensure_index()  # make the CLI/API usable without first opening Streamlit
    state = {
        "question": question,
        "department": department,
        "include_samples": include_samples,
        "search_query": question,
        "retries": 0,
    }
    final, path = dict(state), []

    for update in build_graph().stream(state, stream_mode="updates"):
        for node, changes in update.items():
            path.append(node)
            final.update(changes)

    return {
        "question": question,
        "answer": final.get("answer", ""),
        "route": final.get("route"),
        "path": path,
        "contexts": [d.page_content for d in final.get("documents", [])],
        "sources": final.get("sources", []),
        "ticket_id": final.get("ticket_id"),
    }


def baseline_rag(
    question: str, department: str | None = None, include_samples: bool = True
) -> dict:
    """Plain retrieve -> answer pipeline with no agent logic; the Phase 2 RAGAS baseline."""
    ensure_index()
    docs = retrieve(question, department, include_samples=include_samples)
    answer = ask(ANSWER_PROMPT, context=format_context(docs), question=question)
    return {
        "question": question,
        "answer": answer,
        "contexts": [d.page_content for d in docs],
        "sources": format_sources(docs),
    }


def _print(result: dict) -> None:
    """Pretty-print an agent result in the terminal."""
    print("\nAgent path:", " -> ".join(result["path"]))
    print("Answer:", result["answer"])
    if result["sources"]:
        print("Sources:", ", ".join(result["sources"]))
    if result["ticket_id"]:
        print("Ticket:", result["ticket_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise support agent")
    parser.add_argument("question", nargs="?", help="Omit for interactive mode")
    parser.add_argument("--department", default=None, help="e.g. hr, support")
    parser.add_argument(
        "--no-samples", action="store_true", help="Search uploaded documents only"
    )
    args = parser.parse_args()
    use_samples = not args.no_samples

    if args.question:
        _print(run_agent(args.question, args.department, use_samples))
    else:
        print("Interactive mode. Type 'exit' to quit.")
        while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
            if q:
                _print(run_agent(q, args.department, use_samples))