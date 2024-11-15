import os
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import boto3
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Pinecone and AWS configurations from .env file
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Validate environment variables
if not all([PINECONE_API_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, BUCKET_NAME]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Initialize embedding model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)


class RAGagent:
    """
    RAGagent handles querying Pinecone for relevant embeddings, fetching document data from S3,
    and running the RAG model to generate responses based on user queries.
    """

    def __init__(self, selected_document: str, user_query: str):
        self.selected_document = selected_document
        self.user_query = user_query

    def run(self):
        """
        Main method to execute the RAG pipeline.
        """
        print(f"Running RAG agent for document: {self.selected_document} and query: {self.user_query}")
        try:
            response = self.run_rag_model(self.user_query, self.selected_document)
            return {"answer": response}
        except Exception as e:
            print(f"Error running RAGagent: {e}")
            return {"answer": f"Error: {str(e)}"}

    def run_rag_model(self, user_query, selected_document):
        """
        Executes the RAG pipeline by querying Pinecone and using OpenAI for response generation.
        """
        # Get Pinecone index based on document
        index_name = self.get_index_name_from_document(selected_document)
        pinecone_index = self.get_pinecone_index(index_name)

        # Query Pinecone for relevant embeddings
        pinecone_results = self.query_pinecone(user_query, pinecone_index)
        if pinecone_results == "No matches found.":
            return "No relevant data found in Pinecone."

        # Combine text from top results
        combined_text = "\n".join([result['text'] for result in pinecone_results])

        # Initialize LlamaIndex with combined text
        documents = [SimpleDirectoryReader.from_string(combined_text).load_data()]
        llm = OpenAI(model="text-davinci-003")
        index = VectorStoreIndex.from_documents(documents, llm=llm)
        query_engine = index.as_query_engine()

        # Generate response using LlamaIndex
        response = query_engine.query(user_query)
        return str(response)

    @staticmethod
    def get_index_name_from_document(document_name):
        """Convert document name to the exact Pinecone index name."""
        index_mapping = {
            "ai-and-big-data-in-investments.pdf": "ai-and-big-data-in-investments-index",
            "horan-esg-rf-brief-2022-online.pdf": "horan-esg-rf-brief-2022-online-index",
        }
        if document_name in index_mapping:
            return index_mapping[document_name]
        raise ValueError(f"No matching index found for document: {document_name}")

    def get_pinecone_index(self, index_name: str):
        """
        Retrieves the Pinecone index for the given name.
        """
        existing_indexes = [index.name for index in pinecone_client.list_indexes()]
        if index_name not in existing_indexes:
            raise ValueError(f"Index '{index_name}' does not exist in Pinecone.")
        return pinecone_client.index(index_name)

    def query_pinecone(self, user_query: str, index, top_k=3):
        """
        Queries Pinecone and fetches metadata from S3 if required.
        """
        # Generate query embedding
        query_embedding = sentence_model.encode(user_query).tolist()

        # Query Pinecone index
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Parse results
        results = []
        if response and 'matches' in response:
            for match in response['matches']:
                metadata = match['metadata']
                s3_key = metadata.get("s3_key")

                # Fetch text from S3 using s3_key
                if s3_key:
                    text = self.fetch_from_s3(s3_key)
                else:
                    text = "No text available"
                score = match['score']
                results.append({"text": text, "score": score})
        else:
            return "No matches found."
        return results

    def fetch_from_s3(self, s3_key: str) -> str:
        """
        Fetches text data from S3 using the given key.
        """
        try:
            s3_response = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
            return s3_response["Body"].read().decode("utf-8")
        except Exception as e:
            print(f"Error fetching data from S3 for key {s3_key}: {e}")
            return "Error fetching data from S3."