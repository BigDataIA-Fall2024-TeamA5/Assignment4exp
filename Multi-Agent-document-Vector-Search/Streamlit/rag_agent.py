import os
from dotenv import load_dotenv
from pinecone import Pinecone
import boto3
import openai
import logging

# Load environment variables
load_dotenv()

# API Keys and Environment Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
AI_AND_BIG_DATA_INDEX_NAME = os.getenv("AI_AND_BIG_DATA_INDEX_NAME")
HORAN_ESG_RF_INDEX_NAME = os.getenv("HORAN_ESG_RF_INDEX_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)
openai.api_key = OPENAI_API_KEY

class RAGAgent:
    def __init__(self, document_name, query):
        self.document_name = document_name
        self.query = query

        # Map document names to corresponding Pinecone indexes
        if document_name == "ai-and-big-data-in-investments.pdf":
            self.index_name = AI_AND_BIG_DATA_INDEX_NAME
        elif document_name == "horan-esg-rf-brief-2022-online.pdf":
            self.index_name = HORAN_ESG_RF_INDEX_NAME
        else:
            raise ValueError(f"No index mapped for document: {document_name}")

    def get_or_create_pinecone_index(self, dimension=384):
        """
        Retrieve or create a Pinecone index for the specified document.
        """
        if self.index_name not in pinecone_client.list_indexes().names():
            logging.info(f"Index '{self.index_name}' does not exist. Creating a new index...")
            pinecone_client.create_index(
                name=self.index_name,
                dimension=dimension,  # Dimension for embeddings
                metric='cosine'
            )
            logging.info(f"Index '{self.index_name}' created successfully.")
        else:
            logging.info(f"Index '{self.index_name}' already exists.")
        return pinecone_client.Index(self.index_name)

    def fetch_from_pinecone(self):
        """
        Fetch metadata from Pinecone for the given query.
        """
        index = self.get_or_create_pinecone_index()
        query_vector = [0.0] * 384  # Placeholder for query vector; actual implementation may vary
        response = index.query(
            vector=query_vector, top_k=3, include_metadata=True
        )
        if "matches" in response:
            return response["matches"]
        else:
            return []

    def fetch_text_from_s3(self, s3_key):
        """
        Fetch the text file content from S3 using the provided S3 key.
        """
        try:
            s3_response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
            text = s3_response["Body"].read().decode("utf-8")
            return text
        except Exception as e:
            raise ValueError(f"Error fetching data from S3 for key '{s3_key}': {str(e)}")

    def process_query_with_openai(self, context):
        """
        Process the query using OpenAI's GPT model with the given context.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {self.query}"}
                ],
                max_tokens=500
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise ValueError(f"Error processing query with OpenAI: {str(e)}")

    def run(self):
        """
        Execute the RAG process to retrieve an answer to the query.
        """
        # Fetch matches from Pinecone
        matches = self.fetch_from_pinecone()
        if not matches:
            return {"answer": "No relevant matches found in Pinecone index.", "details": ""}

        # Fetch context from S3 using the S3 key of the top match
        s3_key = matches[0]["metadata"].get("s3_key")
        if not s3_key:
            return {"answer": "No S3 key found in the metadata of the top match.", "details": ""}

        context = self.fetch_text_from_s3(s3_key)

        # Generate the final answer using OpenAI
        answer = self.process_query_with_openai(context)
        return {"answer": answer, "details": context}
