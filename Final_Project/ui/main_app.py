# ui/streamlit_app.py
import streamlit as st
from ui.views import signup_view, login_view, chat_view
from db.memory import init_db

st.set_page_config(page_title="Finance RAG Agent", page_icon="💼", layout="wide")

def init_session_state():
    for k, v in {
        "auth": False,
        "user_id": None,
        "email": "",
        "role": "",
        "history": [],     # short-term memory
        "turns": 0,
        "traces": [],
        "answer": "",
        "session_id": "ui_session"
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main():
    # ✅ Initialize database for long-term memory
    init_db()

    init_session_state()
    st.sidebar.title("Finance RAG Agent")

    if not st.session_state.auth:
        mode = st.sidebar.radio("Navigation", ["Login", "Sign Up"])
        signup_view() if mode == "Sign Up" else login_view()
    else:
        chat_view()

if __name__ == "__main__":
    main()
