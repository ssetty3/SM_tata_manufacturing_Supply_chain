import sqlite3
from helpers import CACHE_STORE
from datetime import datetime
from difflib import SequenceMatcher

DB_PATH = "db/memory.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Main conversation table (stores metadata + text)
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

    # Index for fast filtering by user + role
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_conversations_user_role
    ON conversations(user_id, role);
    """)

    # FTS5 table for text search (only query + answer)
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts 
    USING fts5(query, answer, content='conversations', content_rowid='id');
    """)

    # Sync triggers
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations
    BEGIN
        INSERT INTO conversations_fts(rowid, query, answer)
        VALUES (new.id, new.query, new.answer);
    END;
    """)

    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations
    BEGIN
        UPDATE conversations_fts
        SET query = new.query, answer = new.answer
        WHERE rowid = old.id;
    END;
    """)

    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations
    BEGIN
        DELETE FROM conversations_fts WHERE rowid = old.id;
    END;
    """)

    conn.commit()
    conn.close()





# --- Normalize helper ---
def normalize_text(text: str) -> str:
    return text.strip().lower()

def save_turn(user_id: str, role: str, query: str, answer: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    norm_query = normalize_text(query)

    # 1. Check if a very similar query already exists
    cur.execute("""
        SELECT id, query, answer
        FROM conversations
        WHERE user_id = ? AND role = ?
        ORDER BY timestamp DESC
        LIMIT 10
    """, (user_id, role))
    rows = cur.fetchall()

    for row_id, existing_query, existing_answer in rows:
        score = SequenceMatcher(None, norm_query, normalize_text(existing_query)).ratio()
        if score >= 0.9:  # Duplicate found → skip insert
            conn.close()
            # Still update short-term cache
            cache_key = f"{user_id}:{norm_query}"
            CACHE_STORE[cache_key] = existing_answer
            return

    # 2. Insert new conversation
    cur.execute("""
        INSERT INTO conversations (user_id, role, query, answer, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, role, norm_query, answer, datetime.utcnow()))
    conn.commit()
    conn.close()

    # 3. Update short-term cache
    cache_key = f"{user_id}:{norm_query}"
    CACHE_STORE[cache_key] = answer



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
