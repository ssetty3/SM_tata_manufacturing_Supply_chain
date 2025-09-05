import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import AzureOpenAIEmbeddings
from src.llm_setup import get_groq_llm   # ✅ your Groq LLM wrapper
from src.format_llm_response import pretty_print_result

# Load env
load_dotenv(override=True)


def get_azure_embedding_model():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

# Load embeddings
embeddings = get_azure_embedding_model()

# === Load FAISS index ===
faiss_path = "faiss_index_financial"
vectorstore = FAISS.load_local(
    faiss_path,
    embeddings=embeddings,  # not required for searching
    allow_dangerous_deserialization=True
)

#retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Example: Only retrieve docs for role = "analyst"
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"role": "analyst"}   # 👈 filter here
    }
)

# === Prompt template ===
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

# def standard_rag(query: str):
#     # Step 1: Retrieve docs
#     docs = retriever.invoke(query)
#     context = "\n\n".join([d.page_content for d in docs])

#     # Step 2: Prepare prompt
#     formatted_prompt = prompt.format(question=query, context=context)

#     # Step 3: Call Groq LLM with invoke()
#     llm = get_groq_llm()
#     response = llm.invoke([HumanMessage(content=formatted_prompt)])

#     return response.content, docs


#### Multirole filtering RAG function ####

def standard_rag(query: str, roles: list[str] = None):
    search_kwargs = {"k": 3}

    # If roles are passed, build filter
    if roles:
        # FAISS supports a special syntax using `$in`
        search_kwargs["filter"] = {"role": {"$in": roles}}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Step 1: Retrieve docs
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    # Step 2: Prompt
    formatted_prompt = prompt.format(question=query, context=context)

    # Step 3: LLM call
    llm = get_groq_llm()
    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    return response.content, docs



# === Example usage ===
# if __name__ == "__main__":
#     #query = "What are the key financial risks discussed in these reports?"
#     #query= "what is discussed in the Dividends section of the Deutsche Bank Annual Report 2023?"
#     query= "what was per share dividend in 2023?"
#     answer, docs = standard_rag(query)

#     # Convert LangChain docs → metadata dicts
#     metadata = [
#         {
#             "file": d.metadata.get("file_name", "Unknown"),
#             "role": d.metadata.get("role", "Unknown")
#         }
#         for d in docs
#     ]

#     pretty_print_result(answer, metadata)



if __name__ == "__main__":
    #query = "What are the key financial risks discussed in these reports?"  ## analyst
    #query= "tell me about Tangible Book Value and Average Stock Price per Share 2005–2023 of JPM" ## scientist

    #query = "can you give me bird eye view on the topic: Staying Competitive in the Shrinking Public Markets" ## scientist

    query= "can you give me idea about Overhead ratio of JP Morgan compnay?" ## scientist

    # Example: Fetch only "analyst" + "financial"
    answer, docs = standard_rag(query, roles=["scientist", "financial"])

    metadata = [
        {"file": d.metadata.get("file_name", "Unknown"),
         "role": d.metadata.get("role", "Unknown")}
        for d in docs
    ]
    pretty_print_result(answer, metadata)

    


