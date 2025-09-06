
```
Final_Project/
│
├── src/
│   ├── __init__.py
│   ├── embedding_setup.py
│   ├── faiss_vectorstore.py
│   ├── format_llm_response.py
│   └── llm_setup.py
│
├── helpers.py
├── nodes.py
├── financial_bot.py       # backend agent logic (no UI)
│
├── ui/
│   ├── __init__.py
│   ├── auth.py            # signup/login utilities
│   ├── views.py           # signup_view, login_view, chat_view
│   └── streamlit_app.py   # entry point for Streamlit UI
│
├── users.db               # sqlite db (auto-created)
├── .env
└── requirements.txt
└── README.md

```

## To launch ui

```
python -m streamlit run ui\main_app.py

```