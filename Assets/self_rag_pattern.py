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

# Load FAISS
print("📂 Loading FAISS vectorstore...")
embeddings = get_azure_embedding_model()
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vectorstore loaded successfully.\n")

# === Prompts ===
decision_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an intelligent assistant.  
Decide if retrieval is needed to answer the following question.  

Question: {question}  

Answer only "YES" if retrieval is required, otherwise "NO".
"""
)

reflection_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a helpful assistant.  
Check if the retrieved context is sufficient and relevant.  

Question: {question}  
Context: {context}  

Answer only "YES" if context is useful, otherwise "NO".
"""
)

answer_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a financial assistant.  
Answer the question using the context if available.  

Question:
{question}

Context:
{context}

Answer:
"""
)


def self_rag(user_query: str):
    llm = get_groq_llm()
    print(f"🔍 User Query: {user_query}")

    # Step 1: LLM decides retrieval need
    decision = llm.invoke([HumanMessage(content=decision_prompt.format(question=user_query))]).content.strip()
    print(f"🤖 Retrieval Decision: {decision}")

    context = ""
    docs = []

    if decision.upper() == "YES":
        # Step 2: Retrieve docs
        docs = retriever.invoke(user_query)
        print(f"📑 Retrieved {len(docs)} documents.")
        context = "\n\n".join([d.page_content for d in docs])

        # Step 3: Reflect on context
        reflection = llm.invoke([HumanMessage(content=reflection_prompt.format(
            question=user_query, context=context
        ))]).content.strip()

        print(f"🔎 Context Reflection: {reflection}")

        if reflection.upper() != "YES":
            print("⚠️ Context not useful. Expanding query...")
            expanded_query = f"More detailed explanation about: {user_query}"
            docs = retriever.invoke(expanded_query)
            context = "\n\n".join([d.page_content for d in docs])

    else:
        print("👍 Retrieval skipped, answering directly.")

    # Step 4: Final Answer
    response = llm.invoke([HumanMessage(content=answer_prompt.format(
        question=user_query, context=context
    ))]).content

    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in docs
    ]

    print("✅ Answer generated successfully.\n")
    return response, metadata


# === Example Run ===
if __name__ == "__main__":
    query = "What are common risks mentioned in financial reports?"
    answer, metadata = self_rag(query)

    pretty_print_result(answer, metadata)
