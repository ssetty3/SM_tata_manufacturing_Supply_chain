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

# === Prompts ===
draft_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a financial assistant. 
Using the given context, draft the best possible answer to the question.

Question:
{question}

Context:
{context}

Draft Answer:
"""
)

reflection_prompt = PromptTemplate(
    input_variables=["question", "context", "draft"],
    template="""
You are a critic assistant. 
Review the draft answer for correctness, completeness, and consistency with the context. 
If the draft is good, approve it. If not, refine and improve it.

Question:
{question}

Context:
{context}

Draft Answer:
{draft}

Improved Final Answer:
"""
)


def self_reflective_rag(user_query: str):
    llm = get_groq_llm()
    print(f"🔍 User Query: {user_query}")

    # Step 1: Retrieve context docs
    docs = retriever.invoke(user_query)
    print(f"📑 Retrieved {len(docs)} documents.")
    context = "\n\n".join([d.page_content for d in docs])

    # Step 2: Draft Answer
    print("\n📝 Generating draft answer...")
    draft = llm.invoke([HumanMessage(content=draft_prompt.format(
        question=user_query, context=context
    ))]).content
    print("✅ Draft generated.\n")

    # Step 3: Reflection Step
    print("🤔 Reflecting & improving draft answer...")
    final_answer = llm.invoke([HumanMessage(content=reflection_prompt.format(
        question=user_query, context=context, draft=draft
    ))]).content
    print("✅ Reflection completed.\n")

    # Collect metadata
    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in docs
    ]

    return final_answer, draft, metadata


# === Example Run ===
if __name__ == "__main__":
    query = "What risks are highlighted in the financial report?"
    final_answer, draft, metadata = self_reflective_rag(query)

    print("💡 Draft Answer (Before Reflection):")
    print(draft)
    print("\n💡 Final Answer (After Reflection):")
    pretty_print_result(final_answer, metadata)
