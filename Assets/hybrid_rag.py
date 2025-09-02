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

# === Prompt for query variations ===
fusion_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
Generate 3 alternative search queries that mean the same as the given one. 
Focus on paraphrasing and expanding financial terminology.

Original Query: {query}

Variants:
1.
2.
3.
"""
)

# === Answer Prompt ===
answer_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a financial assistant. 
Use the provided context to answer the user query.

Question:
{question}

Context:
{context}

Answer:
"""
)


def rag_fusion(user_query: str):
    llm = get_groq_llm()
    print(f"🔍 User Query: {user_query}")

    # Step 1: Generate query variations
    fusion_queries = llm.invoke([HumanMessage(content=fusion_prompt.format(query=user_query))]).content
    queries = [q.strip() for q in fusion_queries.split("\n") if q.strip() and not q.strip().isdigit()]
    print("\n🔄 Generated Query Variants:")
    for q in queries:
        print(f" - {q}")

    # Step 2: Retrieve docs for each query
    all_docs = []
    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    # Deduplicate by content
    unique_docs = {d.page_content: d for d in all_docs}.values()
    print(f"\n📑 Retrieved {len(unique_docs)} unique docs across variants.")

    context = "\n\n".join([d.page_content for d in unique_docs])

    # Step 3: Final Answer
    print("\n📝 Generating final fused answer...")
    response = llm.invoke([HumanMessage(content=answer_prompt.format(
        question=user_query, context=context
    ))]).content
    print("✅ Answer generated successfully.\n")

    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in unique_docs
    ]

    return response, metadata


# === Example Run ===
if __name__ == "__main__":
    query = "What is the average stock price from 2005 to 2023?"
    answer, metadata = rag_fusion(query)

    pretty_print_result(answer, metadata)
