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

# === Embeddings & FAISS ===
embeddings = get_azure_embedding_model()
print("📂 Loading FAISS vectorstore...")
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # start small
print("✅ Vectorstore loaded successfully.\n")

# === Prompts ===
draft_answer_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a financial assistant.
Using the context below, draft an initial answer to the question.

Question: {question}
Context:
{context}

Draft Answer:
"""
)

check_missing_prompt = PromptTemplate(
    input_variables=["question", "draft"],
    template="""
You are a self-checking assistant.
Review the draft answer below and check if it fully answers the question.

Question: {question}
Draft Answer: {draft}

If the draft is sufficient → reply "SUFFICIENT".
If important details are missing → reply "MISSING: <what is missing>".
"""
)

refine_answer_prompt = PromptTemplate(
    input_variables=["question", "draft", "context"],
    template="""
You are refining a financial analysis answer.

Original Question: {question}
Previous Draft: {draft}
New Context:
{context}

Refine and improve the answer by incorporating missing details.

Refined Answer:
"""
)

# === Iterative Refinement RAG ===
def iterative_refinement_rag(user_query: str, max_iters: int = 3):
    llm = get_groq_llm()
    print(f"🔍 User Query: {user_query}")

    all_docs = []
    draft = ""
    iteration = 1

    while iteration <= max_iters:
        print(f"\n🔄 Iteration {iteration}...")

        # Step 1: Retrieve new docs (progressively increase k)
        search_kwargs = {"k": 2 + iteration}  # increase retrieval window
        docs = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(user_query)
        all_docs.extend(docs)

        context = "\n\n".join([d.page_content for d in all_docs])

        # Step 2: Generate draft/refined answer
        if iteration == 1:
            formatted_prompt = draft_answer_prompt.format(question=user_query, context=context)
        else:
            formatted_prompt = refine_answer_prompt.format(question=user_query, draft=draft, context=context)

        draft = llm.invoke([HumanMessage(content=formatted_prompt)]).content
        print(f"📝 Draft Answer (iteration {iteration}):\n{draft[:400]}...\n")

        # Step 3: Check sufficiency
        check_response = llm.invoke([HumanMessage(content=check_missing_prompt.format(
            question=user_query, draft=draft
        ))]).content.strip()

        print(f"✅ Sufficiency Check: {check_response}")

        if check_response.startswith("SUFFICIENT"):
            print("🎯 Draft is sufficient. Stopping refinement.")
            break
        else:
            print("⚠️ Draft missing details → Refining with more docs...")
            iteration += 1

    # Step 4: Collect metadata
    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in all_docs
    ]

    return draft, metadata


# === Example Run ===
if __name__ == "__main__":
    query = "What are the main financial risks highlighted by Deutsche Bank?"
    answer, metadata = iterative_refinement_rag(query, max_iters=3)

    pretty_print_result(answer, metadata)
