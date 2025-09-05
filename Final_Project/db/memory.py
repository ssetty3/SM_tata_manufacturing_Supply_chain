import sqlite3
from datetime import datetime

DB_PATH = "db/memory.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        role TEXT,
        query TEXT,
        answer TEXT,
        timestamp DATETIME
    )
    """)
    conn.commit()
    conn.close()

def save_turn(user_id: str, role: str, query: str, answer: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversations (user_id, role, query, answer, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, role, query, answer, datetime.utcnow()))
    conn.commit()
    conn.close()

def load_history(user_id: str, limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT query, answer FROM conversations
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    # return as list of tuples [("user", query), ("assistant", answer)]
    history = []
    for q, a in reversed(rows):
        history.append(("user", q))
        history.append(("assistant", a))
    return history
