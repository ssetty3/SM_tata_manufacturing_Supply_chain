from typing import Dict, Any
from langchain.schema import HumanMessage
from src.llm_setup import get_groq_llm
from helpers import (
    append_trace,
    role_filtered_retriever,
    trim_context,
    CACHE_STORE,
    CONFIG,
)

from langchain.prompts import PromptTemplate

# --- Prompts ---
answer_prompt = PromptTemplate(
    input_variables=["question", "context", "role"],
    template="""
You are a helpful financial assistant for role: {role}.
Use the provided context to answer the user query clearly and concisely.

Question:
{question}

Context:
{context}

Answer:
"""
)

summary_prompt = PromptTemplate(
    input_variables=["history"],
    template="""
You are a summarizer. Summarize the following chat history into a concise but complete memory:

Chat History:
{history}

Summary:
"""
)

check_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a helpful assistant.
Check if the retrieved context is relevant and sufficient to answer the question.

Question: {question}
Context: {context}

Answer only "YES" if context is sufficient, otherwise "NO".
"""
)

# --- Nodes ---
def cache_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    if query in CACHE_STORE:
        cached = CACHE_STORE[query]
        append_trace(state, "cache_check", {"hit": True, "cached_answer": cached})
        state["answer"] = cached
        return state
    append_trace(state, "cache_check", {"hit": False})
    return state

def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    retriever = role_filtered_retriever(state["vectorstore"], state["role"], CONFIG.retriever_k)
    docs = retriever.invoke(state["query"])
    state["docs"] = docs
    state["context"] = trim_context([d.page_content for d in docs], CONFIG.max_context_docs_chars)
    append_trace(state, "retrieve", {"num_docs": len(docs)})
    return state

def relevance_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_groq_llm()
    resp = llm.invoke([HumanMessage(content=check_prompt.format(
        question=state["query"], context=state["context"]
    ))])
    is_relevant = resp.content.strip().upper() == "YES"
    append_trace(state, "relevance_check", {"result": resp.content, "relevant": is_relevant})
    state["context_relevant"] = is_relevant
    return state

def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_groq_llm()
    formatted = answer_prompt.format(
        question=state["query"],
        context=state["context"],
        role=state["role"],
    )
    resp = llm.invoke([HumanMessage(content=formatted)])
    state["answer"] = resp.content
    append_trace(state, "answer", {"answer": resp.content})
    CACHE_STORE[state["query"]] = resp.content
    return state

def summarize_history_node(state: Dict[str, Any]) -> Dict[str, Any]:
    session = state["session"]
    llm = get_groq_llm()
    if session["turns"] % CONFIG.summary_every_n_turns == 0 and session["turns"] > 0:
        history_text = "\n".join(session["history"])
        resp = llm.invoke([HumanMessage(content=summary_prompt.format(history=history_text))])
        session["summary"] = resp.content
        session["history"] = []
        append_trace(state, "summarize", {"summary": resp.content})
    return state


from langchain_community.utilities import SerpAPIWrapper

def internet_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    search = SerpAPIWrapper()
    query = state["query"]

    try:
        results = search.run(query)  # returns a text summary of results
        state["context"] = results
        append_trace(state, "internet_search", {"query": query, "results": results[:500]})
    except Exception as e:
        state["context"] = ""
        append_trace(state, "internet_search", {"query": query, "error": str(e)})

    return state
