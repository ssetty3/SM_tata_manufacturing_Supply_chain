import os, sys, json
from collections import deque
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# === Local imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.llm_setup import get_groq_llm
from src.format_llm_response import pretty_print_result
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore

# === Load env ===
load_dotenv(override=True)

# === Config ===
WORKING_MEMORY_LIMIT = 5
ARCHIVE_FILE = "long_term_memory.json"

# === Memory Stores ===
working_memory = deque(maxlen=WORKING_MEMORY_LIMIT)


# --- Archive Helpers ---
def load_archive():
    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, "r") as f:
            return json.load(f)
    return []


def save_archive(data):
    with open(ARCHIVE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def summarize_and_archive(llm, history):
    """Summarize old turns and save to archive."""
    text = "\n".join(history)
    prompt = f"Summarize the following conversation history:\n{text}\n\nSummary:"
    summary = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    archive = load_archive()
    archive.append(summary)
    save_archive(archive)
    print(f"📝 Archived summary: {summary[:150]}...\n")
    return summary


# === Retrieval + Answer Prompt ===
answer_prompt = PromptTemplate(
    input_variables=["question", "memory", "archives", "context"],
    template="""
You are a helpful assistant. Use memory, archives, and retrieved context.

Conversation Memory:
{memory}

Long-Term Archives:
{archives}

Retrieved Context:
{context}

Question:
{question}

Answer:
"""
)


def long_term_memory_rag(user_query, llm, retriever):
    print(f"\n🔍 User Query: {user_query}")
    working_memory.append(f"User: {user_query}")

    # If memory full → archive
    if len(working_memory) == WORKING_MEMORY_LIMIT:
        print("⚠️ Working memory full → Summarizing and archiving...")
        summarize_and_archive(llm, list(working_memory))
        working_memory.clear()

    # Load archives
    archives = load_archive()
    archive_text = "\n".join(archives) if archives else "No archived summaries yet."

    print(f"🗂️ Working Memory: {list(working_memory)}")
    print(f"📦 Archived Summaries Count: {len(archives)}")

    # Retrieve docs
    docs = retriever.invoke(user_query)
    print(f"📑 Retrieved {len(docs)} documents from vectorstore.")

    context = "\n\n".join([d.page_content for d in docs])

    # Final Answer
    formatted_prompt = answer_prompt.format(
        question=user_query,
        memory="\n".join(working_memory),
        archives=archive_text,
        context=context
    )
    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    # Metadata
    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in docs
    ]

    print("✅ Answer generated successfully.\n")
    pretty_print_result(response.content, metadata)


# === Example Run ===
if __name__ == "__main__":
    llm = get_groq_llm()
    embeddings = get_azure_embedding_model()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Simulated conversation
    queries = [
        "Tell me about Tesla's market position.",
        "How does Tesla compare with BYD?",
        "What are Tesla’s future plans?",
        "Summarize Tesla’s risks.",
        "What’s Tesla’s Q2 revenue?",
        "How is Tesla expanding globally?"
    ]

    for q in queries:
        long_term_memory_rag(q, llm, retriever)
