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

# === Embeddings & Vectorstore ===
embeddings = get_azure_embedding_model()
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === Memory Buffer ===
class ConversationBuffer:
    def __init__(self, window_size=3):
        self.memory = []
        self.window_size = window_size

    def add(self, role, content):
        self.memory.append({"role": role, "content": content})
        # keep only last N turns
        if len(self.memory) > self.window_size * 2:  
            self.memory = self.memory[-(self.window_size * 2):]

    def get_context(self):
        """Format memory for prompt injection."""
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.memory])

# === Prompts ===
rag_prompt = PromptTemplate(
    input_variables=["history", "question", "context"],
    template="""
You are a helpful financial assistant. 
Use the retrieved context and past conversation to answer.

Conversation history:
{history}

User question:
{question}

Retrieved context:
{context}

Answer:
"""
)

# === Conversation Buffer RAG ===
def buffer_memory_rag(user_query: str, memory: ConversationBuffer):
    llm = get_groq_llm()

    # Step 1: Retrieve docs
    docs = retriever.invoke(user_query)
    context = "\n\n".join([d.page_content for d in docs])

    # Step 2: Build prompt with memory
    history = memory.get_context()
    formatted_prompt = rag_prompt.format(
        history=history, question=user_query, context=context
    )

    # Step 3: Get LLM response
    response = llm.invoke([HumanMessage(content=formatted_prompt)]).content

    # Step 4: Save turn into memory
    memory.add("User", user_query)
    memory.add("Assistant", response)

    return response, docs


# === Example Run ===
if __name__ == "__main__":
    memory = ConversationBuffer(window_size=3)

    # Simulate a multi-turn chat
    queries = [
        "Tell me about Tesla’s Q1 2024 results.",
        "What about Q2?",
        "And how did it compare to Ford?",
    ]

    for q in queries:
        answer, docs = buffer_memory_rag(q, memory)
        metadata = [
            {"file": d.metadata.get("file_name", "Unknown"),
             "role": d.metadata.get("role", "Unknown")}
            for d in docs
        ]
        print("\n---")
        pretty_print_result(answer, metadata)
