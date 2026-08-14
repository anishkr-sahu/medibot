"""src/helper.py - Helper functions for PDF processing and embedding initialization."""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_file(data_path: str):
    """Loads all PDF documents from the specified directory path.

    Uses PyPDFLoader and silently skips unreadable/corrupted pages.
    """
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        loader_kwargs={"extract_images": False},
    )
    documents = loader.load()
    return documents


def text_split(extracted_data):
    """Splits document pages into smaller chunks for vector embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_hugging_face_embeddings():
    """Initializes and returns the HuggingFace MiniLM embedding model (384 dimensions)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings