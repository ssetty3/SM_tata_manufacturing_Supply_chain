from typing import Dict, Any, List
from dataclasses import dataclass
from langchain_core.documents import Document

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


def initial_state(user_id: str, session_id: str, role: str, query: str, vectorstore: Any) -> Dict[str, Any]:
    """
    Initialize the full state for a given query run.
    """
    return {
        "query": query,
        "role": role,
        "session": get_session(user_id, session_id),  # <-- unified session logic
        "docs": [],
        "trace": [],
        "vectorstore": vectorstore,   # Required for retrieval
    }


def append_trace(state: Dict[str, Any], step: str, details: Dict[str, Any]):
    """
    Appends a trace log entry to the state for debugging/inspection.
    """
    state["trace"].append({"step": step, "details": details})


from typing import List, Any
from langchain.schema import Document

def role_filtered_retriever(vectorstore: Any, roles: List[str] = None, k: int = 3):
    """
    Wraps a FAISS vectorstore retriever to filter documents by one or more roles.
    
    Args:
        vectorstore: FAISS vectorstore instance.
        roles: List of roles to filter by (e.g., ["scientist", "financial"]).
        k: Number of top documents to retrieve.
    
    Returns:
        A retriever object with an invoke() method that applies role-based filtering.
    """
    search_kwargs = {"k": k}

    if roles:
        # FAISS supports $in filter for metadata
        search_kwargs["filter"] = {"role": {"$in": roles}}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    class RoleRetriever:
        def __init__(self, retriever):
            self.retriever = retriever

        def invoke(self, query: str) -> List[Document]:
            # Use retriever directly; FAISS applies role filter internally
            return self.retriever.invoke(query)

        __call__ = invoke  # alias for direct call

    return RoleRetriever(retriever)



def trim_context(docs_content: List[str], max_chars: int) -> str:
    """
    Joins and trims document contents into a bounded context.
    """
    context = "\n\n".join(docs_content)
    return context[:max_chars]
