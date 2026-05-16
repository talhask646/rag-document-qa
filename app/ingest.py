import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME


def get_embeddings_model():
    """
    Loads the HuggingFace embedding model.
    First call downloads it (~90MB). Subsequent calls load from cache.
    """
    print("Loading embedding model (first run downloads ~90MB)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # small, fast, 384 dimensions
        model_kwargs={"device": "cpu"},  # use CPU (works on any machine)
        encode_kwargs={"normalize_embeddings": True}  # normalise for cosine similarity
    )

    print("Embedding model loaded.")
    return embeddings


def get_pinecone_index():
    """
    Connects to Pinecone and returns the index.
    Creates the index if it doesn't exist yet.
    """
    # Initialise the Pinecone client with our API key
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if our index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,        # must match all-MiniLM-L6-v2 output size
            metric="cosine",      # similarity metric
            spec=ServerlessSpec(  # serverless is the free tier option
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    return pc.Index(PINECONE_INDEX_NAME)


def load_and_chunk_pdf(file_path: str) -> list:
    """
    Loads a PDF and splits it into overlapping chunks.
    Returns a list of LangChain Document objects.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {file_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def ingest_pdf(file_path: str) -> int:
    """
    Full ingestion pipeline: load PDF → chunk → embed → store in Pinecone.
    Returns the number of chunks stored.
    """
    # Step 1: load and chunk the PDF
    chunks = load_and_chunk_pdf(file_path)

    # Step 2: load the embedding model
    embeddings = get_embeddings_model()

    # Step 3: connect to Pinecone
    get_pinecone_index()

    # Step 4: embed all chunks and upload to Pinecone in one call
    # PineconeVectorStore.from_documents() does three things:
    # - calls embeddings.embed_documents() on every chunk
    # - assigns each chunk a unique ID
    # - upserts (insert or update) all vectors into Pinecone
    print("Embedding chunks and uploading to Pinecone...")

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )

    print(f"Successfully stored {len(chunks)} chunks in Pinecone.")
    return len(chunks)


def get_vector_store() -> PineconeVectorStore:
    """
    Returns a PineconeVectorStore connected to our existing index.
    Used by the RAG chain to retrieve chunks at query time.
    """
    embeddings = get_embeddings_model()

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )


# Only runs when executing this file directly
if __name__ == "__main__":
    sample_path = os.path.join("uploads", "sample.pdf")

    if not os.path.exists(sample_path):
        print(f"ERROR: No file found at {sample_path}")
    else:
        # Run the full ingestion pipeline
        count = ingest_pdf(sample_path)

        # Now test retrieval — can we get relevant chunks back?
        print("\n--- Testing retrieval ---")
        vector_store = get_vector_store()

        # similarity_search embeds the query and finds the closest chunks
        test_query = "What is the main topic of this document?"
        results = vector_store.similarity_search(test_query, k=3)

        print(f"Query: '{test_query}'")
        print(f"Top {len(results)} results:\n")

        for i, doc in enumerate(results):
            print(f"Result {i+1} — Page {doc.metadata.get('page', '?')}")
            print(f"Content: {doc.page_content[:200]}...")
            print()