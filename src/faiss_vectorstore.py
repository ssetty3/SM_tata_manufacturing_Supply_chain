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
    faiss_path = "faiss_index_financial"
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings=embeddings,  # not required for searching
        allow_dangerous_deserialization=True
    )
    
    return vectorstore