# visualize_graph.py
import os, sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nodes import (
    cache_check_node,
    retrieve_node,
    relevance_check_node,
    answer_node,
    summarize_history_node,
)

load_dotenv(override=True)

def build_workflow():
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

    return workflow

if __name__ == "__main__":
    workflow = build_workflow()
    graph = workflow.compile()   # ✅ must compile first

    print("\n=== Workflow (Mermaid Format) ===")
    print(graph.get_graph().draw_mermaid())  # ✅ use graph.get_graph()

    # Optionally save PNG (requires Graphviz)
    try:
        graph.get_graph().draw_png("agent_graph.png")
        print("✅ Saved agent_graph.png in current folder.")
    except Exception as e:
        print(f"⚠️ Could not save PNG: {e}")
