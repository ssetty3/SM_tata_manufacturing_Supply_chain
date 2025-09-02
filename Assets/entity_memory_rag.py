import json, os
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_setup import get_groq_llm
from src.embedding_setup import get_azure_embedding_model
from src.faiss_vectorstore import get_vectorstore

# === Entity Memory Store (JSON) ===
ENTITY_FILE = "entity_memory.json"

def load_entities():
    if os.path.exists(ENTITY_FILE):
        with open(ENTITY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_entities(data):
    with open(ENTITY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_entity_memory(entity, fact):
    memory = load_entities()
    if entity not in memory:
        memory[entity] = []
    memory[entity].append(fact)
    save_entities(memory)

def get_entity_facts(entity):
    memory = load_entities()
    return memory.get(entity, [])

# === Init ===
embeddings = get_azure_embedding_model()
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === Prompt Templates ===
extract_entities_prompt = PromptTemplate(
    input_variables=["text"],
    template="Extract the key entities (companies, people, orgs) mentioned in the following text:\n{text}\n\nEntities:"
)

answer_prompt = PromptTemplate(
    input_variables=["question", "entity_facts", "context"],
    template="""
You are a financial assistant. Use both entity facts and retrieved docs to answer.

Entity Memory:
{entity_facts}

User Question:
{question}

Retrieved Context:
{context}

Answer:
"""
)

def entity_memory_rag(user_query, llm):
    # Step 1: Extract entities
    extraction = llm.invoke([HumanMessage(content=extract_entities_prompt.format(text=user_query))])
    entities = [e.strip() for e in extraction.content.split(",") if e.strip()]
    print(f"🔎 Extracted Entities: {entities}")

    # Step 2: Retrieve facts for entities
    entity_facts = []
    for ent in entities:
        facts = get_entity_facts(ent)
        if facts:
            entity_facts.extend([f"{ent}: {f}" for f in facts])

    # Step 3: Retrieve docs
    docs = retriever.invoke(user_query)
    context = "\n\n".join([d.page_content for d in docs])

    # Step 4: Generate final answer
    formatted_prompt = answer_prompt.format(
        question=user_query,
        entity_facts="\n".join(entity_facts) if entity_facts else "No prior facts",
        context=context
    )
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    answer = response.content.strip()

    # Step 5: Store new entity facts (for persistence)
    for ent in entities:
        update_entity_memory(ent, f"Asked: {user_query} | Answered: {answer[:100]}...")

    return answer, docs


# === Example Run ===
if __name__ == "__main__":
    llm = get_groq_llm()

    q1 = "Tell me about Tesla's growth strategy."
    a1, _ = entity_memory_rag(q1, llm)
    print("\n🔹 Answer1:", a1)

    q2 = "What do you know about Tesla?"
    a2, _ = entity_memory_rag(q2, llm)
    print("\n🔹 Answer2:", a2)
