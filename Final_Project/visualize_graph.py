# visualize_graph.py
import os
import sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Adjust import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your nodes
from nodes import (
    cache_check_node,
    retrieve_node,
    relevance_check_node,
    internet_search_node,
    answer_node,
    summarize_history_node,
)

# Load env if needed
load_dotenv(override=True)


def build_workflow():
    workflow = StateGraph(dict)   # ✅ flat dict state

    # Add nodes
    workflow.add_node("cache_check", cache_check_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("relevance", relevance_check_node)
    workflow.add_node("internet_search", internet_search_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("summarize", summarize_history_node)

    # Edges
    workflow.set_entry_point("cache_check")
    workflow.add_edge("cache_check", "retrieve")
    workflow.add_edge("retrieve", "relevance")

    workflow.add_conditional_edges(
        "relevance",
        lambda s: "answer" if s.get("context_relevant") else "internet_search",
        {
            "answer": "answer",
            "internet_search": "internet_search",
        },
    )

    workflow.add_edge("internet_search", "answer")
    workflow.add_edge("answer", "summarize")
    workflow.add_edge("summarize", END)

    return workflow


if __name__ == "__main__":
    workflow = build_workflow()
    graph = workflow.compile()

    # Print Mermaid code
    mermaid_code = graph.get_graph().draw_mermaid()
    print("\n=== Workflow (Mermaid Format) ===\n")
    print(mermaid_code)

    # Try saving PNG (requires Graphviz installed)
    try:
        graph.get_graph().draw_png("agent_graph.png")
        print("\n✅ Saved agent_graph.png in current folder.")
    except Exception as e:
        print(f"\n⚠️ Could not save PNG: {e}")
