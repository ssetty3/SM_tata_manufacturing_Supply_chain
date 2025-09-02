import os, sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore
from src.format_llm_response import pretty_print_result
from helpers import get_session, initial_state
from nodes import (
    cache_check_node,
    retrieve_node,
    relevance_check_node,
    answer_node,
    summarize_history_node,
)

# --- Load environment and FAISS ---
load_dotenv(override=True)
embeddings = get_azure_embedding_model()
vectorstore = get_vectorstore()

# --- Graph Setup ---
workflow = StateGraph(dict)

workflow.add_node("cache_check", cache_check_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("relevance", relevance_check_node)
workflow.add_node("answer", answer_node)
workflow.add_node("summarize", summarize_history_node)

workflow.set_entry_point("cache_check")
workflow.add_edge("cache_check", "retrieve")
workflow.add_edge("retrieve", "relevance")
workflow.add_conditional_edges(
    "relevance",
    lambda s: "answer" if s.get("context_relevant") else "answer",
    {"answer": "answer"},
)
workflow.add_edge("answer", "summarize")
workflow.add_edge("summarize", END)

graph = workflow.compile()


# --- Run Agent ---
def run_agent(user_id: str, session_id: str, role: str, query: str):
    state = initial_state(user_id, session_id, role, query, vectorstore)
    result = graph.invoke(state)

    session = result["session"]
    session["turns"] += 1
    session["history"].append(f"Q: {query}\nA: {result['answer']}")

    return result

# --- Example ---
if __name__ == "__main__":
    query = "What risks are mentioned in Deutsche Bank annual report?"
    result = run_agent("user1", "sess1", role="analyst", query=query)

    pretty_print_result(result["answer"], [
        {"file": d.metadata.get("file_name", "Unknown"), "role": d.metadata.get("role", "Unknown")}
        for d in result["docs"]
    ])

    print("\n=== Trace ===")
    for step in result["trace"]:
        print(step)
