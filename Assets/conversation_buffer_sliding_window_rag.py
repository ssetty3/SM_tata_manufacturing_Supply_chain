from collections import deque
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_setup import get_groq_llm
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore

class ConversationBuffer:
    def __init__(self, window_size=5):
        """
        Keep last N turns of conversation.
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, role, content):
        self.buffer.append({"role": role, "content": content})
        print(f"📝 Added turn ({role}): {content[:60]}...")

    def get_context(self):
        return "\n".join([f"{t['role']}: {t['content']}" for t in self.buffer])


# === Setup RAG ===
embeddings = get_azure_embedding_model()
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

answer_prompt = PromptTemplate(
    input_variables=["history", "question", "context"],
    template="""
You are a financial assistant. Use the conversation history + retrieved docs to answer.

Conversation history:
{history}

Question:
{question}

Context:
{context}

Answer:
"""
)

def buffer_rag(user_query, memory, llm):
    # Step 1: Add user turn
    memory.add("user", user_query)

    # Step 2: Retrieve documents
    docs = retriever.invoke(user_query)
    context = "\n\n".join([d.page_content for d in docs])

    # Step 3: Prepare prompt with buffer
    history = memory.get_context()
    formatted_prompt = answer_prompt.format(
        history=history,
        question=user_query,
        context=context
    )

    # Step 4: LLM response
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    answer = response.content.strip()

    # Step 5: Add assistant turn
    memory.add("assistant", answer)

    return answer, docs


# === Example Run ===
if __name__ == "__main__":
    llm = get_groq_llm()
    memory = ConversationBuffer(window_size=4)  # keep last 4 turns

    query1 = "Tell me about Tesla."
    answer1, docs1 = buffer_rag(query1, memory, llm)
    print("\n🔹 Answer1:", answer1)

    query2 = "What about their latest revenue?"
    answer2, docs2 = buffer_rag(query2, memory, llm)
    print("\n🔹 Answer2:", answer2)

    query3 = "And who are their competitors?"
    answer3, docs3 = buffer_rag(query3, memory, llm)
    print("\n🔹 Answer3:", answer3)

    print("\n=== Current Conversation Buffer ===")
    print(memory.get_context())
