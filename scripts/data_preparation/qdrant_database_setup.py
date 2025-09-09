from qdrant_client import QdrantClient
from qdrant_client.http import models
# Create (or recreate) collection
import os
from dotenv import load_dotenv

# === Load Azure credentials ===
load_dotenv(override=True)

def qdrant_connection(qdrant_url,qdrant_api_key):
    qdrant_client = QdrantClient(
        url= qdrant_url, 
        api_key=qdrant_api_key,
    )
    return qdrant_client




def create_collection(collection_name, qdrant_client):
  # Define vector size based on your embedding model (e.g., Azure OpenAI = 1536)
  VECTOR_SIZE = 3072

  qdrant_client.create_collection(
      collection_name=collection_name,
      vectors_config=models.VectorParams(
          size=VECTOR_SIZE,
          distance=models.Distance.COSINE  # or .DOT if needed
      )
  )
  print(f"✅ Collection '{collection_name}' created successfully.")

def delete_collection(collection_name,qdrant_client):
    qdrant_client.delete_collection(collection_name)
    print(f"🗑️ Collection '{collection_name}' deleted successfully.")


def delete_document(collection_name,qdrant_client, document_id):
    # Delete by ID
    qdrant_client.delete(collection_name=collection_name, points_selector=[document_id])
    print(f"🗑️ Document with ID '{document_id}' deleted successfully.")






if __name__=="__main__":
    qdrant_url= os.getenv("QDRANT_URL")
    qdrant_api_key=os.getenv("QDRANT_API_KEY")

    collection_name = "financial_reports_collection"
    qdrant_client= qdrant_connection(qdrant_url,qdrant_api_key)
    print(qdrant_client.get_collections())
    create_collection(collection_name,qdrant_client)
    delete_collection(collection_name,qdrant_client)
    delete_document(collection_name, qdrant_client, "doc123")

