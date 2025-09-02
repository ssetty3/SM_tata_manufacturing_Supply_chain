import os, sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_setup import get_groq_llm
from src.format_llm_response import pretty_print_result
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore

# === Load env ===
load_dotenv(override=True)

# === Config ===
CONVERSATION_MEMORY = []

# --- Fake Structured Knowledge (could be DB/API) ---
structured_knowledge = {
    "Tesla": {
        "latest_revenue": "25B USD (Q2 2025)",
        "ceo": "Elon Musk",
        "hq": "Austin, Texas",
    },
    "BYD": {
        "latest_revenue": "18B USD (Q2 2025)",
        "ceo": "Wang Chuanfu",
        "hq": "Shenzhen, China",
    }
}

# === Prompts ===
answer_prompt = PromptTemplate(
    input_variables=["question", "memory", "structured", "context"],
    template="""
You are a financial assistant. Use the following information:

Conversation Memory:
{memory}

Structured Knowledge (trusted facts):
{structured}

Retrieved Context:
{context}

Question:
{question}

Answer:
"""
)

def knowledge_injected_rag(user_query, llm, retriever):
    print(f"\n🔍 User Query: {user_query}")

    # Append to memory
    CONVERSATION_MEMORY.append(f"User: {user_query}")

    # Select structured data if entity match
    structured_context = "No structured data found."
    for company, facts in structured_knowledge.items():
        if company.lower() in user_query.lower():
            structured_context = "\n".join([f"{k}: {v}" for k,v in facts.items()])
            break

    # Retrieve docs
    docs = retriever.invoke(user_query)
    print(f"📑 Retrieved {len(docs)} documents from FAISS.")

    context = "\n\n".join([d.page_content for d in docs])

    # Final Answer
    formatted_prompt = answer_prompt.format(
        question=user_query,
        memory="\n".join(CONVERSATION_MEMORY[-5:]),  # last 5 turns
        structured=structured_context,
        context=context
    )

    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in docs
    ]

    print("✅ Answer generated successfully.\n")
    pretty_print_result(response.content, metadata)


# === Example Run ===
if __name__ == "__main__":
    llm = get_groq_llm()
    embeddings = get_azure_embedding_model()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    queries = [
        "What is Tesla’s latest revenue?",
        "Who is BYD’s CEO?",
        "Compare Tesla and BYD market presence."
    ]

    for q in queries:
        knowledge_injected_rag(q, llm, retriever)
