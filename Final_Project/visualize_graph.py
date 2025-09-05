# Adjust import path
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from financial_bot import get_graph


if __name__ == "__main__":
    graph = get_graph()

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
