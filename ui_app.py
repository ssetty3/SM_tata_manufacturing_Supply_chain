import streamlit as st
from typing import List
import sqlite3
import threading
import random
import time

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- import nodes and utilities ---
from scripts.vectorstore.faiss_vectorstore import get_retriever
from scripts.rag_agent.nodes import (
    question_rewriter, question_classifier, off_topic_response, retrieval_grader,
    generate_answer, refine_question, search_internet,
    AgentState, on_topic_router, proceed_router
)
from scripts.rag_agent.chains import rag_chain, off_topic_chain, internet_helper_chain

# --- Setup ---
rag_chain = rag_chain()
off_topic_rag_chain = off_topic_chain()
internet_chain = internet_helper_chain()

sqlite_conn = sqlite3.connect(
    r"C:\Users\smm931389\Desktop\RAG_patterns\scripts\rag_agent\main_financebot.sqlite",
    check_same_thread=False,
)
checkpointer = SqliteSaver(sqlite_conn)


# Friendly labels for node names (unused for fake loader, kept for future)
NODE_LABELS = {
    "question_rewriter": "✏️ Rewriting question",
    "question_classifier": "📊 Classifying topic",
    "retrieve": "📂 Retrieving documents",
    "retrieval_grader": "⚖️ Grading relevance",
    "refine_question": "🔧 Refining question",
    "generate_answer": "🧠 Generating answer",
    "search_internet": "🌐 Searching internet",
    "off_topic_response": "💡 Handling off-topic",
}


def get_compiled_workflow_graph(roles: List[str]):
    retriever = get_retriever(roles=roles)
    workflow = StateGraph(AgentState)

    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("off_topic_response", off_topic_response)

    def retrieve_node(state: AgentState):
        documents = retriever.invoke(state["rephrased_question"])
        state["documents"] = documents
        return state

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("retrieval_grader", retrieval_grader)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("search_internet", search_internet)

    workflow.add_edge("question_rewriter", "question_classifier")
    workflow.add_conditional_edges(
        "question_classifier",
        on_topic_router,
        {"retrieve": "retrieve", "off_topic_response": "off_topic_response"},
    )
    workflow.add_edge("retrieve", "retrieval_grader")
    workflow.add_conditional_edges(
        "retrieval_grader",
        proceed_router,
        {
            "generate_answer": "generate_answer",
            "refine_question": "refine_question",
            "search_internet": "search_internet",
        },
    )
    workflow.add_edge("refine_question", "retrieve")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("search_internet", END)
    workflow.add_edge("off_topic_response", END)

    workflow.set_entry_point("question_rewriter")
    graph = workflow.compile(checkpointer=checkpointer)
    return graph


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="FinanceBot", page_icon="🤖", layout="wide")
st.title("🤖 FinanceBot – RAG Assistant")

# Session state for authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "role" not in st.session_state:
    st.session_state["role"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---- Login / Signup ----
if not st.session_state["authenticated"]:
    st.subheader("🔑 Login / Signup")
    with st.form("login_form"):
        role = st.text_input("Enter your role (e.g., analyst, manager)").strip().lower()
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if role and username and password:  # Simple auth placeholder
            st.session_state["authenticated"] = True
            st.session_state["role"] = role
            st.success(f"✅ Logged in as {username} ({role})")
            st.rerun()
        else:
            st.error("❌ Please enter role, username and password.")

else:
    st.sidebar.success(f"Logged in as: {st.session_state['role']}")

    # Build the graph (per role)
    graph = get_compiled_workflow_graph(roles=[st.session_state["role"]])

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input box
    if user_query := st.chat_input("Ask me anything about finance..."):
        # Add user msg to session
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Assistant response container
        with st.chat_message("assistant"):
            placeholder = st.empty()
            progress_placeholder = st.empty()

            # Mutable container for thread result / error
            result_container = {"state": None, "error": None}

            # Define background worker for graph invocation
            def run_graph_worker():
                try:
                    input_data = {"question": HumanMessage(content=user_query)}
                    result = graph.invoke(
                        input=input_data, config={"configurable": {"thread_id": 1}}
                    )
                    result_container["state"] = result
                except Exception as e:
                    result_container["error"] = f"{type(e).__name__}: {str(e)}"

            # Start graph in background thread
            worker_thread = threading.Thread(target=run_graph_worker, daemon=True)
            worker_thread.start()

            # Progress bar behaviour
            progress_bar = progress_placeholder.progress(0)
            progress = 0
            while worker_thread.is_alive():
                increment = random.randint(1, 3)
                progress = min(progress + increment, 90)
                progress_bar.progress(progress)
                time.sleep(0.12)

            # Ensure worker has finished
            worker_thread.join(timeout=0.1)

            # Handle error
            if result_container["error"]:
                progress_bar.progress(100)
                progress_placeholder.empty()
                st.error(f"Error while running graph: {result_container['error']}")
                # stop execution for this query
                st.stop()

            # Finalize progress
            progress_bar.progress(100)
            time.sleep(0.2)
            progress_placeholder.empty()

            # Render final answer
            result_state = result_container["state"]
            if result_state:
                ai_msg = None
                for msg in result_state["messages"]:
                    if isinstance(msg, AIMessage):
                        ai_msg = msg
                if ai_msg:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": ai_msg.content}
                    )
                    placeholder.markdown(ai_msg.content)
                else:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": str(result_state)}
                    )
                    placeholder.markdown("Done. (No AIMessage found in state.)")
            else:
                st.error("No result returned from the workflow.")

