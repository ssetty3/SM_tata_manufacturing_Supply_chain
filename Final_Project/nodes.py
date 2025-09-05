from typing import Dict, Any
from langchain.schema import HumanMessage
import re
from src.llm_setup import get_groq_llm
from helpers import (
    append_trace,
    role_filtered_retriever,
    trim_context,
    CACHE_STORE,
    CONFIG,
)
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SerpAPIWrapper

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
    session = state.get("session", {})
    user_id = session.get("user_id", "default")

    # Create unique cache key per user + query
    cache_key = f"{user_id}:{query}"

    if cache_key in CACHE_STORE:
        cached = CACHE_STORE[cache_key]
        append_trace(state, "cache_check", {"hit": True, "cached_answer": cached})
        state["answer"] = cached
        return state

    append_trace(state, "cache_check", {"hit": False})
    return state


# Detect chit-chat queries (simple regex for demo)
CHITCHAT_PATTERNS = re.compile(r"^(hi|hello|hey|how are you|what's up|who are you)", re.I)

def chitchat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    if CHITCHAT_PATTERNS.match(query.strip()):
        llm = get_groq_llm()
        resp = llm.invoke([HumanMessage(content=f"You are a friendly assistant. Reply casually to: {query}")])
        state["answer"] = resp.content
        append_trace(state, "chitchat", {"answer": resp.content})
        CACHE_STORE[query] = resp.content
    return state

# def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     retriever = role_filtered_retriever(state["vectorstore"], state["role"], CONFIG.retriever_k)
#     docs = retriever.invoke(state["query"])
#     state["docs"] = docs
#     state["context"] = trim_context([d.page_content for d in docs], CONFIG.max_context_docs_chars)
#     append_trace(state, "retrieve", {"num_docs": len(docs)})
#     return state


def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    retriever = role_filtered_retriever(state["vectorstore"], state["role"], CONFIG.retriever_k)
    docs = retriever.invoke(state["query"])
    state["docs"] = docs

    context_lines = []

    for doc in docs:
        lines = doc.page_content.splitlines()
        table_rows = []
        header_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Count numbers in the line
            numbers = [w for w in stripped.split() if w.replace(".", "").replace(",", "").isdigit()]
            
            if len(numbers) >= 2:
                # Numeric row: separate columns by |
                table_rows.append(" | ".join(stripped.split()))
            else:
                # Non-numeric line: treat as header if table_rows empty, else as paragraph
                if table_rows and header_lines:
                    context_lines.append("\n".join(header_lines))
                    header_lines = []
                header_lines.append(stripped)

        # Append table if exists
        if table_rows:
            if header_lines:
                # Markdown header for table
                header = table_rows[0]
                separator = " | ".join(["---"] * len(header.split("|")))
                context_lines.append(f"| {header} |")
                context_lines.append(f"| {separator} |")
                context_lines.extend([f"| {row} |" for row in table_rows[1:]])
            else:
                context_lines.extend(table_rows)
        # Append leftover headers/text
        context_lines.extend(header_lines)

    # Limit to max context length
    state["context"] = "\n".join(context_lines)[:CONFIG.max_context_docs_chars]
    append_trace(state, "retrieve", {"num_docs": len(docs)})

    # Debug print
    #print("=== Retrieved Context (processed) ===")
    #print(state["context"][:500])
    #print("===================================")

    return state





def relevance_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_groq_llm()
    resp = llm.invoke([HumanMessage(content=check_prompt.format(
        question=state["query"], context=state["context"]
    ))])
    is_relevant = resp.content.strip().upper() == "YES"

    # Only store the boolean
    state["context_relevant"] = is_relevant

    append_trace(state, "relevance_check", {
        "raw_llm_output": resp.content.strip(),
        "relevant": is_relevant
    })
    return state



# def answer_node_stream(state: Dict[str, Any]):
#     """Stream answer tokens incrementally into state['answer']."""    
#     llm = get_groq_llm()

#     prompt = f"""
#     You are a helpful financial assistant for role: {state['role']}.
#     Use the provided context to answer clearly.

#     Question:
#     {state['query']}

#     Context:
#     {state.get('context', '')}

#     Answer:
#     """

#     chunks = []
#     for chunk in llm.stream([HumanMessage(content=prompt)]):
#         delta = chunk.content or ""
#         if delta:
#             chunks.append(delta)
#             partial_answer = "".join(chunks)

#             # Yield a partial state update (LangGraph expects dicts with updates)
#             yield {
#                 "answer": partial_answer,
#                 "trace": [
#                     {"step": "answer_stream", "details": {"token": delta}}
#                 ]
#             }

#     # Finalize
#     final_answer = "".join(chunks)
#     state["answer"] = final_answer
#     append_trace(state, "answer", {"answer": final_answer})
#     CACHE_STORE[state["query"]] = final_answer

#     # Yield the final completed state
#     yield {"answer": final_answer}


def answer_node_stream(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate answer incrementally and store final result in state['answer']."""    
    llm = get_groq_llm()

    prompt = f"""
You are a helpful financial assistant for role: {state['role']}.
Use the provided context to answer clearly.

Question:
{state['query']}

Context:
{state.get('context', '')}

Answer:
"""

    # Stream tokens to console for live feedback
    final_answer = ""
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        delta = chunk.content or ""
        if delta:
            final_answer += delta
            # Print live streaming to terminal
            print(delta, end="", flush=True)

    # Save final answer in state
    state["answer"] = final_answer
    append_trace(state, "answer", {"answer": final_answer})
    CACHE_STORE[state["query"]] = final_answer

    # Print newline after streaming
    print("\n")

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
