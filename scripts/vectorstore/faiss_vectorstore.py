from langchain_community.vectorstores import FAISS
from src.embedding_setup import get_azure_embedding_model
from dotenv import load_dotenv

def get_vectorstore():
    """Load the FAISS vector store."""
   
    # Load environment variables
    load_dotenv(override=True)

    # Load embeddings
    embeddings = get_azure_embedding_model()

    # === Load FAISS index ===
    faiss_path = r"C:\Users\smm931389\Desktop\RAG_patterns\faiss_index_financial"
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings=embeddings,  # not required for searching
        allow_dangerous_deserialization=True
    )
    
    return vectorstore


def get_retriever(roles):
    """Load the retriever."""
    vectorstore = get_vectorstore()


    search_kwargs = {"k": 3}
    if roles:
        # FAISS supports $in filter for metadata
        search_kwargs["filter"] = {"role": {"$in": roles}}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    return retriever