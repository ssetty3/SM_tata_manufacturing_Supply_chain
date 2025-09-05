# ui/stream_runner.py
from typing import Generator, Tuple, Dict, Any
from helpers import initial_state
from financial_bot import get_graph, load_backends

def stream_run(query: str, role: str, user_id: str) -> Tuple[Generator[str, None, None], Dict[str, Any]]:
    """
    Run the graph with streaming support.
    Returns:
      - generator for streaming tokens (usable with st.write_stream)
      - final state containing traces/docs/etc.
    """
    _, vectorstore = load_backends()
    graph = get_graph()

    # Initial state
    state = initial_state(user_id, "ui_session", role, query, vectorstore)
    state.setdefault("docs", [])
    state.setdefault("trace", [])

    # --- Define generator for token streaming ---
    def token_generator() -> Generator[str, None, None]:
        for msg, metadata in graph.stream(state, stream_mode="messages"):
            if msg.content:  # each msg is a HumanMessage/AIMessage
                yield msg.content  # streamable text chunk

    # After streaming finishes, get final state
    final_state = graph.invoke(state)

    return token_generator, final_state
