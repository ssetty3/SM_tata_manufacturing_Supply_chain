import os, sys, hashlib
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Allow importing src helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_setup import get_groq_llm
from src.format_llm_response import pretty_print_result
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore

load_dotenv(override=True)

# -------------------------
# Prompts
# -------------------------
rewrite_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a financial query rewriting assistant.\n"
        "Rewrite the following user query into a more specific, detailed financial analysis query.\n\n"
        "Original query:\n{query}\n\n"
        "Rewritten query:"
    )
)

answer_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are a concise financial assistant. Use the provided context to answer precisely.\n\n"
        "Question:\n{question}\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    )
)

cross_check_prompt = PromptTemplate(
    input_variables=["question", "contexts"],
    template=(
        "You are performing cross-checking across multiple retrieved contexts.\n"
        "Synthesize a single, best answer. If there is disagreement or uncertainty across contexts, "
        "explicitly call it out and explain briefly.\n\n"
        "Question:\n{question}\n\n"
        "Combined Contexts:\n{contexts}\n\n"
        "Final, cross-checked answer:"
    )
)

# -------------------------
# Utilities
# -------------------------
def _hash_doc(d) -> str:
    file_name = d.metadata.get("file_name", "unknown")
    key = f"{file_name}||{d.page_content[:3000]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def _truncate_text(text: str, max_chars: int = 8000) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n\n...[truncated]"

def _collect_metadata(docs):
    return [
        {
            "file": d.metadata.get("file_name", "Unknown"),
            "role": d.metadata.get("role", "Unknown"),
            "snippet": (d.page_content[:300] + "...") if len(d.page_content) > 300 else d.page_content
        }
        for d in docs
    ]

# -------------------------
# Self-Consistency RAG
# -------------------------
def self_consistency_rag(user_query: str, k_per_variant: int = 3, use_cross_check: bool = True, roles: list | None = None):
    print("🔁 Initializing models and vectorstore...")
    embeddings = get_azure_embedding_model()
    vectorstore = get_vectorstore()
    llm = get_groq_llm()
    print("✅ Ready.\n")

    search_kwargs = {"k": k_per_variant}
    if roles:
        search_kwargs["filter"] = {"role": {"$in": roles}}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Variant A: Original
    print("\n--- Variant A: ORIGINAL ---")
    docs_original = retriever.invoke(user_query)
    print(f"   • Retrieved {len(docs_original)} docs")

    # Variant B: Rewritten
    print("\n--- Variant B: LLM-REWRITTEN ---")
    rewritten_q = llm.invoke([HumanMessage(content=rewrite_prompt.format(query=user_query))]).content.strip()
    print(f"   • Rewritten query: {rewritten_q}")
    docs_rewritten = retriever.invoke(rewritten_q)
    print(f"   • Retrieved {len(docs_rewritten)} docs")

    # Variant C: Expanded
    print("\n--- Variant C: EXPANDED ---")
    expanded_q = f"Provide detailed financial insights (drivers, risks, metrics, trends) about: {user_query}"
    print(f"   • Expanded query: {expanded_q}")
    docs_expanded = retriever.invoke(expanded_q)
    print(f"   • Retrieved {len(docs_expanded)} docs")

    # Deduplication
    all_docs, seen = [], set()
    for bundle in (docs_original, docs_rewritten, docs_expanded):
        for d in bundle:
            h = _hash_doc(d)
            if h not in seen:
                seen.add(h)
                all_docs.append(d)

    print(f"\n🧹 Deduplicated to {len(all_docs)} unique docs")

    # Merge context
    merged_context = "\n\n".join([d.page_content for d in all_docs])
    merged_context = _truncate_text(merged_context, max_chars=8000)

    # Cross-check or simple answer
    if use_cross_check:
        print("\n🧪 Cross-checking with merged contexts...")
        prompt_text = cross_check_prompt.format(question=user_query, contexts=merged_context)
    else:
        print("\n📝 Using merged context directly...")
        prompt_text = answer_prompt.format(question=user_query, context=merged_context)

    response = llm.invoke([HumanMessage(content=prompt_text)])
    answer = response.content

    metadata = _collect_metadata(all_docs)

    print("\n✅ Self-consistency RAG complete.\n")
    return answer, metadata

# -------------------------
# Example Run
# -------------------------
if __name__ == "__main__":
    query = "What are the key financial risks in Tesla’s 2023 annual report?"
    answer, metadata = self_consistency_rag(query, k_per_variant=3, use_cross_check=True)

    pretty_print_result(answer, metadata)
