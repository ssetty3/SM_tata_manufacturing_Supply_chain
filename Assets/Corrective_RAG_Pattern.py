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

embeddings = get_azure_embedding_model()

# === Load FAISS index ===
print("📂 Loading FAISS vectorstore...")
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vectorstore loaded successfully.\n")

# === Answering Prompt ===
answer_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a financial assistant. Use the provided context to answer the user query.

Question:
{question}

Context:
{context}

Answer:
"""
)

# === Corrective Check Prompt ===
check_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a helpful assistant. 
Check if the retrieved context is relevant and sufficient to answer the question.

Question: {question}
Context: {context}

Answer only "YES" if context is sufficient, otherwise "NO".
"""
)


def corrective_rag(user_query: str):
    llm = get_groq_llm()
    print(f"🔍 User Query: {user_query}")

    # Step 1: Retrieve docs
    docs = retriever.invoke(user_query)
    print(f"📑 Retrieved {len(docs)} documents.")

    for i, d in enumerate(docs, 1):
        print(f"   └─ Doc {i}: {d.metadata.get('file_name','Unknown')} "
              f"(role={d.metadata.get('role','Unknown')})")

    context = "\n\n".join([d.page_content for d in docs])

    # Step 2: Check relevance
    print("\n🤖 Checking relevance of retrieved docs...")
    check_response = llm.invoke([
        HumanMessage(content=check_prompt.format(
            question=user_query, context=context
        ))
    ]).content.strip()

    print(f"✅ Relevance Check Result: {check_response}")

    # Step 3: If context not sufficient → expand query
    if check_response.upper() != "YES":
        print("⚠️ Context insufficient. Expanding query...")
        expanded_query = f"Provide detailed financial insights about: {user_query}"
        print(f"🔄 Expanded Query: {expanded_query}")

        docs = retriever.invoke(expanded_query)
        print(f"📑 Retrieved {len(docs)} documents after expansion.")

        for i, d in enumerate(docs, 1):
            print(f"   └─ Expanded Doc {i}: {d.metadata.get('file_name','Unknown')} "
                  f"(role={d.metadata.get('role','Unknown')})")

        context = "\n\n".join([d.page_content for d in docs])
    else:
        print("👍 Context is sufficient, proceeding with answer generation.")

    # Step 4: Final Answer
    print("\n📝 Generating final answer...")
    formatted_prompt = answer_prompt.format(question=user_query, context=context)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    print("✅ Answer generated successfully.\n")
    return response.content, docs


# === Example Run ===
if __name__ == "__main__":
    #query = "What risks are mentioned?"
    #query= "who is the CEO of Deutsche Bank in 2023?"
    query = "who is the spider man of Deutsche Bank?"
    answer, docs = corrective_rag(query)

    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in docs
    ]

    pretty_print_result(answer, metadata)
