from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_setup import get_groq_llm
from src.format_llm_response import pretty_print_result
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore
from langchain_tavily import TavilySearch

# === Load environment ===
load_dotenv(override=True)

embeddings = get_azure_embedding_model()

# === Load FAISS index ===
print("📂 Loading FAISS vectorstore...")
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vectorstore loaded successfully.\n")


# === Tavily Setup ===
tavily_search = TavilySearch(k=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))

# === Prompts ===
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

    context = "\n\n".join([d.page_content for d in docs])

    # Step 2: Check relevance
    print("\n🤖 Checking relevance of retrieved docs...")
    check_response = llm.invoke([
        HumanMessage(content=check_prompt.format(
            question=user_query, context=context
        ))
    ]).content.strip()

    print(f"✅ Relevance Check Result: {check_response}")

    unified_metadata = []

    # Step 3: If not relevant → Web search fallback
    if check_response.upper() != "YES":
        print("⚠️ Context insufficient → Falling back to Web Search (Tavily)...")

        try:
            search_results = tavily_search.invoke(user_query)  # Tavily returns list[str]
           
            print("\n🔎 Raw Tavily Response:")
            print(search_results)  # full dump for debugging
        except Exception as e:
            print(f"❌ Tavily search failed: {e}")
            search_results = ["No results from Tavily due to error."]

        context_parts = []
        for idx, snippet in enumerate(search_results, start=1):
            # Ensure we treat non-strings safely
            snippet = str(snippet)

            if "error" in snippet.lower():
                print(f"🌐 Web Result {idx}: ❌ Tavily error → {snippet}")
                continue  # don't include "error" into context
            else:
                print(f"🌐 Web Result {idx}:")
                print(f"   🔹 Snippet: {snippet[:200]}...\n")

            context_parts.append(snippet)
            unified_metadata.append({
                "file": f"🌐 WebResult-{idx}",
                "role": "web",
                "url": "N/A"
            })

        if context_parts:
            context = "\n\n".join(context_parts)
        else:
            context = "No useful web results available."
            print("⚠️ No valid web snippets found, passing fallback message to LLM.")

        print(f"🌐 Processed {len(context_parts)} valid web results from Tavily.")

    else:
        print("👍 Context is sufficient, proceeding with answer generation.")
        for d in docs:
            unified_metadata.append({
                "file": d.metadata.get("file_name", "Unknown"),
                "role": d.metadata.get("role", "Unknown"),
                "url": "N/A"
            })

    # Step 4: Final Answer
    print("\n📝 Generating final answer...")
    formatted_prompt = answer_prompt.format(question=user_query, context=context)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    print("✅ Answer generated successfully.\n")
    return response.content, unified_metadata



# === Example Run ===
if __name__ == "__main__":
    query = "What is Tesla’s latest quarterly revenue?"
    answer, metadata = corrective_rag(query)

    pretty_print_result(answer, metadata)
