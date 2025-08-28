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

# 🔹 What is Self-Consistency RAG?
 
### The Self-Consistency RAG pattern improves reliability by running multiple retrieval pathways (e.g., original query, rewritten query, expanded query). It collects results from all variants, deduplicates them, and lets the LLM cross-check across contexts before answering.

### This reduces hallucinations and ensures the answer is consistent, even if one retrieval path fails.



```

User Query
    │
    ▼
 ┌───────────────┐
 │ Variant 1     │  (Original Query Retrieval)
 └───────────────┘
    │
    ▼
 ┌───────────────┐
 │ Variant 2     │  (LLM-Rewritten Query)
 └───────────────┘
    │
    ▼
 ┌───────────────┐
 │ Variant 3     │  (Expanded Query Retrieval)
 └───────────────┘
    │
    ▼
 Merge & Deduplicate Docs
    │
    ▼
Check Consistency / Relevance
    ├── Consistent & Relevant → Answer with Context
    └── Inconsistent / Insufficient → Cross-Check or Retry with Web Search
    │
    ▼
  Final LLM Answer
    │
    ▼
Pretty Print (with Metadata)



```

# 🔹 Iterative Refinement RAG (a.k.a. Chain of Density / Progressive Summarization)

## 📌 Definition

### Instead of fetching all documents at once and answering, the LLM iteratively refines its answer by pulling in more documents step by step.

### First pass → draft answer from initial small context (e.g., top-k=2).

### Second pass → identify gaps/uncertainties in the draft.

### Retrieve more docs → refine.

### Repeat until confident or max iterations.

### This balances efficiency (not loading all docs) with coverage (fills missing details progressively).

## ✅ When to Use

### When you have long context windows but want to avoid overloading the LLM at once.

### When queries require multi-step reasoning (e.g., risk analysis, regulatory summaries, historical trends).

### When you want progressive, more confident answers instead of one-shot retrieval.

```
User Query
    │
    ▼
Retriever (small top-k, e.g., 2)
    │
    ▼
Draft Answer (LLM)
    │
    ▼
Identify Missing Info (LLM self-check)
    │
    ├── If sufficient → return Final Answer
    │
    └── If insufficient →
          Refine Query / Retrieve More Docs →
          Update Answer →
          Loop (max N iterations)


```


# 🔹 Self-Reflective RAG (a.k.a. Critic–Corrector RAG)

## 📝 Definition:

### In Self-Reflective RAG, after generating an answer, the LLM itself (or a second LLM/agent) acts as a critic: it reviews the answer for accuracy, completeness, and consistency against the retrieved context. If issues are found, it self-corrects before returning the final response.

### Think of it as: 
### LLM generates → LLM critiques → LLM improves → Final Answer

## ✅ When to use:

- When accuracy & trustworthiness are critical (finance, legal, healthcare).

- When the retrieved documents are dense or ambiguous.

- When you want hallucination reduction beyond simple context checks.

- Useful if you cannot guarantee perfect retrieval, but still want reliable answers.

```
User Query
    │
    ▼
Retriever (FAISS / Vector DB)
    │
    ▼
Generate Draft Answer (LLM)
    │
    ▼
Critic / Self-Reflection Step (LLM)
    ├── If Accurate → Final Answer
    └── If Issues Found → Refine & Correct → Return Improved Answer


```

# 🔹 Chain-of-Thought RAG (a.k.a. Step-by-Step Reasoning RAG)

## 📖 Definition

### Instead of giving an answer directly, the LLM is prompted to reason step by step over the retrieved context.
### This encourages more structured answers and reduces hallucinations, since the model explicitly shows its reasoning.

## ✅ When to Use

    - When answers require multi-step reasoning (e.g., "compare risks across companies").

    - When you want the LLM’s thought process visible for debugging / transparency.

    - Good for auditing correctness since intermediate reasoning is logged.

```
User Query
    │
    ▼
Retriever (FAISS / Vector DB)
    │
    ▼
LLM (Reason Step-by-Step with Context)
    │
    ▼
Final Answer (with reasoning trace available)


```