import sqlite3
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.llm_setup import get_groq_llm
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore

# === Persistent DB Setup ===
DB_PATH = "conversation_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_turn(role, content):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO memory (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def fetch_recent_turns(n=5):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM memory ORDER BY id DESC LIMIT ?", (n,))
    rows = cur.fetchall()
    conn.close()
    return "\n".join([f"{r}: {c}" for r, c in rows[::-1]])  # reverse chronological

# === Init ===
init_db()
embeddings = get_azure_embedding_model()
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === Prompt Template ===
answer_prompt = PromptTemplate(
    input_variables=["memory", "question", "context"],
    template="""
You are a financial assistant. Use both the stored conversation memory 
and retrieved documents to answer.

Past Memory:
{memory}

User Question:
{question}

Retrieved Context:
{context}

Answer:
"""
)

def persistent_rag(user_query, llm):
    # Step 1: Fetch stored memory
    memory_text = fetch_recent_turns(n=5)

    # Step 2: Retrieve docs
    docs = retriever.invoke(user_query)
    context = "\n\n".join([d.page_content for d in docs])

    # Step 3: Build prompt
    formatted_prompt = answer_prompt.format(
        memory=memory_text,
        question=user_query,
        context=context
    )

    # Step 4: Get response
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    answer = response.content.strip()

    # Step 5: Save this turn into DB
    save_turn("user", user_query)
    save_turn("assistant", answer)

    return answer, docs


# === Example Run ===
if __name__ == "__main__":
    llm = get_groq_llm()

    q1 = "Hi, I am an investor interested in Tesla."
    a1, _ = persistent_rag(q1, llm)
    print("\n🔹 Answer1:", a1)

    q2 = "What did I just ask you about?"
    a2, _ = persistent_rag(q2, llm)
    print("\n🔹 Answer2:", a2)

    q3 = "Now tell me about their competitors."
    a3, _ = persistent_rag(q3, llm)
    print("\n🔹 Answer3:", a3)
