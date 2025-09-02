from typing import Dict, Any
from dataclasses import dataclass

# === In-memory cache ===
CACHE_STORE: Dict[str, str] = {}

# === Session Management ===
SESSIONS: Dict[str, Dict[str, Any]] = {}


@dataclass
class Config:
    retriever_k: int = 3                # top-k docs for retriever
    max_context_docs_chars: int = 4000  # trim context length
    summary_every_n_turns: int = 3      # summarize history every N turns


CONFIG = Config()


def get_session(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Returns or initializes a session for a given user/session_id.
    """
    key = f"{user_id}:{session_id}"
    if key not in SESSIONS:
        SESSIONS[key] = {
            "summary": "",
            "history": [],
            "turns": 0,
        }
    return SESSIONS[key]


def initial_state(user_id: str, session_id: str, role: str, query: str, vectorstore) -> Dict[str, Any]:
    """
    Initializes the state dictionary passed through LangGraph.
    """
    session = get_session(user_id, session_id)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "role": role,
        "query": query,
        "vectorstore": vectorstore,
        "docs": [],
        "context": "",
        "answer": "",
        "trace": [],
        "session": session,
    }


def append_trace(state: Dict[str, Any], step: str, details: Dict[str, Any]):
    """
    Appends a trace log entry to the state for debugging/inspection.
    """
    state["trace"].append({"step": step, "details": details})


def role_filtered_retriever(vectorstore, role: str, k: int):
    """
    Wraps FAISS retriever to filter documents by role metadata.
    """

    class RoleRetriever:
        def __init__(self, retriever, role):
            self.retriever = retriever
            self.role = role

        def invoke(self, query: str):
            docs = self.retriever.invoke(query)
            return [d for d in docs if d.metadata.get("role") == self.role]

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return RoleRetriever(base_retriever, role)



def trim_context(docs_content: list, max_chars: int) -> str:
    """
    Joins and trims document contents into a bounded context.
    """
    context = "\n\n".join(docs_content)
    return context[:max_chars]
