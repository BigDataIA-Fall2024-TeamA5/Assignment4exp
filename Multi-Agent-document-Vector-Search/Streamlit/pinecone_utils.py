import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Fetch API key and environment from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
AI_AND_BIG_DATA_INDEX = os.getenv("AI_AND_BIG_DATA_INDEX_NAME")
HORAN_ESG_RF_INDEX = os.getenv("HORAN_ESG_RF_INDEX_NAME")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))  # Default dimension set to 384 for your indexes

# Initialize the Pinecone client
pinecone_client = Pinecone(api_key=pinecone_api_key)

# Function to retrieve a specific Pinecone index
def get_pinecone_index(index_name):
    """
    Retrieve or create a Pinecone index by name.
    """
    if index_name not in pinecone_client.list_indexes():
        # Specify serverless spec with cloud provider and region
        pinecone_client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=pinecone_environment
            )
        )
    return pinecone_client.Index(index_name)

# Function to dynamically select the index based on document name
def select_index(document_name):
    """
    Map document names to the appropriate Pinecone index.
    """
    index_mapping = {
        "ai-and-big-data-in-investments.pdf": AI_AND_BIG_DATA_INDEX,
        "horan-esg-rf-brief-2022-online.pdf": HORAN_ESG_RF_INDEX,
    }
    if document_name in index_mapping:
        return get_pinecone_index(index_mapping[document_name])
    raise ValueError(f"No index mapping found for document: {document_name}")

# Function to store embeddings in Pinecone
def store_embeddings(document_name, metadata, embeddings):
    """
    Store embeddings in the Pinecone index corresponding to the document.
    """
    index = select_index(document_name)
    document_id = metadata.get("id", "unknown_id")  # Use 'id' from metadata or fallback to 'unknown_id'
    index.upsert(vectors=[(document_id, embeddings, metadata)])

# Function to retrieve embeddings from Pinecone
def query_embeddings(document_name, query_vector, top_k=3):
    """
    Query embeddings from the Pinecone index corresponding to the document.
    """
    index = select_index(document_name)
    response = index.query(top_k=top_k, vector=query_vector, include_metadata=True)
    if response and 'matches' in response:
        return response['matches']
    return []

# Optional function to delete all data from a specific Pinecone index
def delete_index_data(index_name):
    """
    Delete all data from the specified Pinecone index.
    """
    index = get_pinecone_index(index_name)
    index.delete(delete_all=True)

# Optional function to list all indexes
def list_all_indexes():
    """
    List all available indexes in Pinecone.
    """
    return pinecone_client.list_indexes()
