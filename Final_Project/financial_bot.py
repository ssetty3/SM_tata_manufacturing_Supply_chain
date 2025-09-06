import os
import sys
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any
import operator

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    answer_node_stream,
    chitchat_node,
    summarize_history_node,
    CHITCHAT_PATTERNS,
)

# --- Load environment and FAISS ---
load_dotenv(override=True)
embeddings = get_azure_embedding_model()
vectorstore = get_vectorstore()

def load_backends():
    """Expose embeddings and vectorstore for other modules (like UI)."""
    return embeddings, vectorstore

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

# Nodes
workflow.add_node("cache_check", cache_check_node)
workflow.add_node("chitchat", chitchat_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("relevance", relevance_check_node)
workflow.add_node("internet_search", internet_search_node)
workflow.add_node("answer_stream", answer_node_stream)
workflow.add_node("summarize", summarize_history_node)

# Flow
workflow.set_entry_point("cache_check")

# --- Conditional branching after cache_check ---
def route_from_cache(state: BotState) -> str:
    query = state["query"].strip()
    # Case 1: cache hit → go straight to answer
    if "answer" in state and state["answer"]:
        return "answer_stream"
    # Case 2: chit-chat query → go to chit-chat
    if CHITCHAT_PATTERNS.match(query):
        return "chitchat"
    # Case 3: normal → go to retrieval pipeline
    return "retrieve"

workflow.add_conditional_edges(
    "cache_check",
    route_from_cache,
    {"answer_stream": "answer_stream", "chitchat": "chitchat", "retrieve": "retrieve"},
)

# If chit-chat handled, go straight to answer_stream
workflow.add_edge("chitchat", "answer_stream")

# Retrieval pipeline
workflow.add_edge("retrieve", "relevance")
workflow.add_conditional_edges(
    "relevance",
    lambda s: "answer_stream" if s.get("context_relevant") else "internet_search",
    {"answer_stream": "answer_stream", "internet_search": "internet_search"},
)
workflow.add_edge("internet_search", "answer_stream")

# Final steps
workflow.add_edge("answer_stream", "summarize")
workflow.add_edge("summarize", END)

graph = workflow.compile()

def get_graph():
    """Expose the compiled LangGraph workflow to UI or other scripts."""
    return graph

# --- Run Agent ---
def run_agent(user_id: str, session_id: str, role: str, query: str):
    state = initial_state(user_id, session_id, role, query, vectorstore)
    result = graph.invoke(state)

    # Update session after final answer
    session = result["session"]
    session["turns"] += 1
    session["history"].append(f"Q: {query}\nA: {result.get('answer', '')}")

    return result

# --- Example Execution ---
if __name__ == "__main__":
    query= "what is discussed in the Dividends section of the Deutsche Bank Annual Report 2023?"  #-> analyst
    #query= "tell me about Tangible Book Value and Average Stock Price per Share 2005–2023 of JPM"  #> scientist
    #query= "explain JOURNEY TO THE CLOUD of JPM" # scientist

    #query = "What is the average stock price from 2005 to 2023?"
    #query = "Tell me about the financial performance of Microsoft in 2025."
    result = run_agent("user1", "sess1", role="scientist", query=query)

    pretty_print_result(
        result.get("answer", ""),
        [
            {
                "file": doc.metadata.get("file_name", "Unknown"),
                "role": doc.metadata.get("role", "Unknown"),
            }
            for doc in result.get("docs", [])
        ],
        traces=result.get("trace", []),
    )
