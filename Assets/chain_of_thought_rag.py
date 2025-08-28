from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_setup import get_groq_llm
from src.format_llm_response import pretty_print_result
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore

# === Load environment ===
load_dotenv(override=True)

# Load FAISS index
print("📂 Loading FAISS vectorstore...")
embeddings = get_azure_embedding_model()
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vectorstore loaded successfully.\n")

# === Prompt with Step-by-Step Reasoning ===
cot_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a financial reasoning assistant. 
Use the provided context to carefully reason step by step, then provide the final answer.

Question:
{question}

Context:
{context}

Reasoning (step by step):
1. 

Final Answer:
"""
)


def cot_rag(user_query: str):
    llm = get_groq_llm()
    print(f"🔍 User Query: {user_query}")

    # Step 1: Retrieve docs
    docs = retriever.invoke(user_query)
    print(f"📑 Retrieved {len(docs)} documents.")
    context = "\n\n".join([d.page_content for d in docs])

    # Step 2: Ask LLM with CoT prompt
    print("\n🧠 Generating step-by-step reasoning...")
    response = llm.invoke([HumanMessage(content=cot_prompt.format(
        question=user_query, context=context
    ))]).content
    print("✅ Answer generated with reasoning.\n")

    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in docs
    ]

    return response, metadata


# === Example Run ===
if __name__ == "__main__":
    query = "Compare liquidity risks of Deutsche Bank vs Unilever."
    answer, metadata = cot_rag(query)

    pretty_print_result(answer, metadata)
