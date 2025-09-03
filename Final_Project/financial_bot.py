import os
import sys
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any
import operator

from langgraph.graph import StateGraph, END
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore
from src.format_llm_response import pretty_print_result
from helpers import initial_state
from nodes import (
    cache_check_node,
    retrieve_node,
    relevance_check_node,
    internet_search_node,
    answer_node,
    summarize_history_node,
)

# --- Load environment and FAISS ---
load_dotenv(override=True)
embeddings = get_azure_embedding_model()
vectorstore = get_vectorstore()

# --- Define TypedDict State with accumulators ---
class BotState(TypedDict, total=False):
    query: str
    role: str
    vectorstore: Any
    docs: Annotated[List[Any], operator.add]
    trace: Annotated[List[Dict[str, Any]], operator.add]
    context: str
    context_relevant: bool
    answer: str
    session: Dict[str, Any]

# --- Build workflow ---
workflow = StateGraph(BotState)

workflow.add_node("cache_check", cache_check_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("relevance", relevance_check_node)
workflow.add_node("internet_search", internet_search_node)
workflow.add_node("answer", answer_node)
workflow.add_node("summarize", summarize_history_node)

workflow.set_entry_point("cache_check")
workflow.add_edge("cache_check", "retrieve")
workflow.add_edge("retrieve", "relevance")
workflow.add_conditional_edges(
    "relevance",
    lambda s: "answer" if s.get("context_relevant") else "internet_search",
    {"answer": "answer", "internet_search": "internet_search"},
)
workflow.add_edge("internet_search", "answer")
workflow.add_edge("answer", "summarize")
workflow.add_edge("summarize", END)

graph = workflow.compile()

# --- Run Agent ---
def run_agent(user_id: str, session_id: str, role: str, query: str):
    # Initialize state with correct keys
    state = initial_state(user_id, session_id, role, query, vectorstore)
    # Make sure we're providing the accumulators
    state.setdefault("docs", [])
    state.setdefault("trace", [])

    result = graph.invoke(state)

    session = result["session"]
    session["turns"] += 1
    session["history"].append(f"Q: {query}\nA: {result['answer']}")

    return result

# --- Example Execution ---
if __name__ == "__main__":
    query = "Tell me about the financial performance of Microsoft in 2025."
    result = run_agent("user1", "sess1", role="analyst", query=query)

    pretty_print_result(
        result["answer"],
        [
            {
                "file": doc.metadata.get("file_name", "Unknown"),
                "role": doc.metadata.get("role", "Unknown"),
            }
            for doc in result["docs"]
        ],
        traces=result["trace"],
    )

