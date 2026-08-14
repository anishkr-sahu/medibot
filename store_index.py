import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.helper import (
    download_hugging_face_embeddings,
    load_pdf_file,
    text_split,
)

# 1. Load variables from .env file into os.environ
load_dotenv()

# 2. Retrieve API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Safety Check: Verify key is loaded before initializing
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Check your .env file!")

# 3. Extract, Chunk, and Embed Data
extracted_data = load_pdf_file("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# 4. Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# 5. Create index safely (only if it doesn't exist)
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists.")

# 6. Embed each chunk and upsert the embeddings into Pinecone
print("Upserting documents to Pinecone vector store...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks, index_name=index_name, embedding=embeddings
)
print("Successfully stored embeddings in Pinecone!")