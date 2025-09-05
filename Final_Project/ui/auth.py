# ui/auth.py
import sqlite3
import bcrypt
import os

DB_PATH = os.getenv("APP_DB_PATH", "users.db")

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            role TEXT NOT NULL
        );
    """)
    return conn

def create_user(email: str, password: str, role: str) -> bool:
    conn = _connect()
    try:
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        conn.execute("INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
                     (email.lower().strip(), pw_hash, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(email: str, password: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, email, password_hash, role FROM users WHERE email=?", (email.lower().strip(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    user_id, email, pw_hash, role = row
    if bcrypt.checkpw(password.encode("utf-8"), pw_hash):
        return {"user_id": str(user_id), "email": email, "role": role}
    return None
