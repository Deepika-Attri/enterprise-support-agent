"""Tool layer: a small SQLite company database plus order lookup and ticket creation."""

import sqlite3
from contextlib import contextmanager

from src.config import settings


# Open a database connection, commit on success and always close it afterwards
@contextmanager
def db():
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Create the orders and tickets tables and seed three sample orders if the table is empty
def init_db() -> None:
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY, customer_name TEXT, item TEXT,
                status TEXT, expected_delivery TEXT, amount_inr INTEGER);
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, reason TEXT,
                status TEXT DEFAULT 'open', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)
        if conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO orders VALUES (?,?,?,?,?,?)",
                [
                    (
                        "ORD-1001",
                        "Aarav Sharma",
                        "Wireless Earbuds",
                        "Shipped",
                        "2026-10-06",
                        2499,
                    ),
                    (
                        "ORD-1002",
                        "Priya Singh",
                        "Running Shoes",
                        "Delivered",
                        "2026-09-28",
                        3299,
                    ),
                    (
                        "ORD-1003",
                        "Rohan Gupta",
                        "Smart Watch",
                        "Processing",
                        "2026-10-10",
                        5999,
                    ),
                ],
            )


# Tool 1: look up an order by ID (case-insensitive), returns None if it doesn't exist
def get_order(order_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id.upper(),)
        ).fetchone()
    return dict(row) if row else None


# Tool 2: open a human-handoff ticket and return its ID like TKT-1001
def create_ticket(question: str, reason: str) -> str:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO tickets (question, reason) VALUES (?, ?)", (question, reason)
        )
        return f"TKT-{1000 + cur.lastrowid}"
