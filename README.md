# RAG_patterns


# Query Rewrite RAG

## This pattern is used when:

### The user’s query is ambiguous, too short, or needs reformulation (e.g., “What about DB risk?” → rewrite into “What are Deutsche Bank’s key financial risks mentioned in the annual report?”).

### We want to improve retrieval by expanding/rephrasing the query before hitting the retriever.

```

 User Query ──► LLM Query Rewriter ──► Enhanced Query ──► Retriever ──► Docs
                                                      │
                                                      ▼
                                             Context + User Query ──► LLM

```

.

# 🔹 What is Corrective RAG?

## This pattern is useful when the retriever brings back irrelevant or insufficient context.

### The pipeline checks the quality of retrieved docs and either:

### Proceeds with the answer if context is sufficient ✅ Or triggers a correction step (e.g., expand query, rerun retrieval, or fallback to web search) ❌

```
User Query
    │
    ▼
Retriever (FAISS / Vector DB)
    │
    ▼
Check Relevance (LLM or Heuristic)
    ├── Relevant → Use Context → Answer with LLM
    └── Not Relevant → Correct Query / Expand Search → Retry Retrieval → Answer
```