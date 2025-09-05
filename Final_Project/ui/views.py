# ui/views.py
import streamlit as st
from ui.auth import create_user, authenticate_user
from ui.stream_runner import stream_run
from db.memory import load_history, save_turn   # ✅ long-term memory functions


# ----------------- SIGNUP -----------------
def signup_view():
    st.markdown("## ✍️ Create an account")

    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["analyst", "scientist", "manager"])
        submitted = st.form_submit_button("Sign Up")

    if submitted:
        if not email or not password:
            st.error("Email and password are required.")
            return

        ok = create_user(email, password, role)
        if ok:
            st.success("✅ Signup successful. Please log in.")
        else:
            st.error("❌ Email already exists.")


# ----------------- LOGIN -----------------
def login_view():
    st.markdown("## 🔑 Log in")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")

    if submitted:
        user = authenticate_user(email, password)
        if user:
            st.session_state.auth = True
            st.session_state.user_id = user["user_id"]
            st.session_state.email = user["email"]
            st.session_state.role = user["role"]

            # ✅ Load long-term memory from DB
            history = load_history(user["user_id"])
            st.session_state.history = history if history else []

            st.session_state.setdefault("traces", [])
            st.session_state.setdefault("turns", 0)

            st.success(f"Welcome, {user['email']} ({user['role']})")
            st.rerun()
        else:
            st.error("❌ Invalid credentials.")


# ----------------- CHAT -----------------
def chat_view():
    st.sidebar.markdown(f"**👤 Logged in as:** {st.session_state.email}")
    st.sidebar.markdown(f"**🛠 Role:** `{st.session_state.role}`")
    st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.update({"auth": False}))

    st.title("💼 Finance RAG Agent")
    left, right = st.columns([0.65, 0.35])

    with left:
        st.subheader("💬 Chat")
        for sender, msg in st.session_state.history:
            with st.chat_message("user" if sender == "user" else "assistant"):
                st.markdown(msg)

        prompt = st.chat_input("Ask your financial question...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.history.append(("user", prompt))

            gen, state = stream_run(prompt, st.session_state.role, st.session_state.user_id)

            with st.chat_message("assistant"):
                final_answer = st.write_stream(gen())

            st.session_state.history.append(("assistant", final_answer))
            st.session_state.traces = state.get("trace", [])
            st.session_state.turns += 1

            # ✅ save with role also
            save_turn(st.session_state.user_id, st.session_state.role, prompt, final_answer)
    
    with right:
        st.subheader("📝 Run Details")
        traces = st.session_state.get("traces", [])
        if traces:
            for t in traces:
                st.markdown(f"**Step:** `{t.get('step')}`")
                st.json(t.get("details", {}))
                st.divider()
        else:
            st.info("No traces yet. Ask a question to see details.")