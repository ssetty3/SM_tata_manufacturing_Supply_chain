from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_setup import get_groq_llm   # ✅ your Groq LLM wrapper
from src.format_llm_response import pretty_print_result
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore

# load environment variables
load_dotenv(override=True)

# Load embeddings
embeddings = get_azure_embedding_model()

# Ensure the vectorstore is loaded
vectorstore = get_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === Prompt for rewriting the query ===
rewrite_template = """
You are a query rewriting assistant.
Rewrite the following user query into a more specific and detailed financial analysis query.

Original query:
{query}

Rewritten query:
"""

rewrite_prompt = PromptTemplate(
    input_variables=["query"],
    template=rewrite_template
)


# === Main Answering Prompt ===
template = """
You are a financial assistant. Use the provided context to answer the user query.

Question:
{question}

Context:
{context}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template
)



def query_rewrite_rag(user_query: str, roles: list[str] = None):
    llm = get_groq_llm()

    # Step 1: Rewrite query
    formatted_prompt = rewrite_prompt.format(query=user_query)
    rewritten_query = llm.invoke([HumanMessage(content=formatted_prompt)]).content
    print(f"🔄 Rewritten Query: {rewritten_query}\n")

    # Step 2: Apply role filter if given
    search_kwargs = {"k": 3}
    if roles:
        search_kwargs["filter"] = {"role": {"$in": roles}}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Step 3: Retrieve docs
    docs = retriever.invoke(rewritten_query)
    context = "\n\n".join([d.page_content for d in docs])

    # Step 4: Ask final LLM
    formatted_prompt = prompt.format(question=rewritten_query, context=context)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    return response.content, docs


if __name__ == "__main__":
    query = "DB risk?"
    answer, docs = query_rewrite_rag(query, roles=["analyst"])

    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in docs
    ]

    pretty_print_result(answer, metadata)

