import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import AzureOpenAIEmbeddings
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
from tqdm import tqdm
import uuid

# === Load env vars ===
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# === Config ===
PDF_DIR = "./financial_pdfs"
COLLECTION_NAME = "financial_reports_collection"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 350
VECTOR_DIM = 3072  # or whatever your model outputs

# === PDF file mapping ===
pdf_files = {
    "JPM_Annual_2023.pdf": "https://www.jpmorganchase.com/.../annualreport-2023.pdf",
    "JPM_Annual_2024.pdf": "https://www.sec.gov/.../annualreport-2024.pdf",
    "DB_Annual_2023.pdf": "https://investor-relations.db.com/.../20-F-2023.pdf",
    "JPM_SE_Annual_2023.pdf": "https://www.jpmorgan.com/.../2023-annual-report-english.pdf",
    "Unilever_Annual_2024.pdf": "https://www.unilever.com/.../20-f-2024.pdf",
}

# === Qdrant connection ===
def qdrant_connection(qdrant_url, qdrant_api_key):
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# === Load PDFs ===
def load_pdfs(pdf_dir):
    docs = []
    print("📂 Loading PDFs...")
    for fname, url in pdf_files.items():
        path = os.path.join(pdf_dir, fname)
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            pages = loader.load()
            for page in pages:
                page.metadata["source_url"] = url
                page.metadata["filename"] = fname
            docs.extend(pages)
            print(f"✅ '{fname}' → {len(pages)} pages")
        else:
            print(f"⚠️ '{fname}' not found; skipping.")
    return docs

# === Split documents ===
def split_docs(docs):
    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"🔢 Total chunks: {len(chunks)}")
    return chunks

# === Embed and store ===
def embed_and_store(chunks, client):
    embeddings = AzureOpenAIEmbeddings(
        deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        model=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE
    )

    # Create collection if not exists
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )

    print("📥 Uploading chunks to Qdrant...")
    points = []

    for chunk in tqdm(chunks, desc="Indexing Chunks"):
        vector = embeddings.embed_query(chunk.page_content)
        metadata = chunk.metadata

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text_content": chunk.page_content,
                    "URL_path": metadata.get("source_url"),
                    "page_number": metadata.get("page", -1),
                    "pdf_name": metadata.get("filename")
                }
            )
        )

    # Upload to Qdrant
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Uploaded {len(points)} chunks to collection '{COLLECTION_NAME}'")

# === Main ingest ===
def ingest():
    docs = load_pdfs(PDF_DIR)
    if not docs:
        print("❌ No PDFs loaded — check PDF_DIR and filenames.")
        return

    chunks = split_docs(docs)
    qdrant_client = qdrant_connection(QDRANT_URL, QDRANT_API_KEY)
    embed_and_store(chunks, qdrant_client)

if __name__ == "__main__":
    ingest()
