import streamlit as st
import boto3
from dotenv import load_dotenv
import os
from PIL import Image
import io
import base64
from urllib.parse import urlparse
from rag_agent import RAGagent
from arxiv_agent import ArxivAgent
from web_search_agent import WebSearchAgent

# Load environment variables
load_dotenv()

# S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PDFS_FOLDER = "pdfs/"  # Folder in the S3 bucket containing PDFs

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Set up Streamlit page configuration with a wide layout
st.set_page_config(page_title="PDF Research Application", layout="wide")

# Initialize session state variables
if 'pdf_data' not in st.session_state:
    st.session_state['pdf_data'] = []
if 'selected_pdf' not in st.session_state:
    st.session_state['selected_pdf'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to fetch PDFs and metadata from S3
def fetch_pdf_data_from_s3():
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PDFS_FOLDER)
        pdf_files = []
        for obj in response.get('Contents', []):
            file_key = obj['Key']
            if file_key.endswith('.pdf'):
                file_name = file_key.split('/')[-1]
                pdf_files.append({
                    "title": file_name,
                    "link": f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_key}",
                    "brief_summary": "No summary available",
                    "image_link": None  # Placeholder for potential thumbnail
                })
        return pdf_files
    except Exception as e:
        st.error(f"Error fetching PDF data from S3: {e}")
        return []

# Research interface
def extract_document_name(url):
    """Extract the document name from an S3 URL."""
    return os.path.basename(urlparse(url).path)


def research_interface():
    st.title("Multi-Agent Research Interface")

    # Ensure PDF data is loaded
    if not st.session_state['pdf_data']:
        st.error("No PDF data available. Please load your documents.")
        return

    # Document selection
    st.subheader("Document Selection")

    # Ensure PDF data is loaded
    if not st.session_state['pdf_data']:
        st.error("No PDF data available. Please load your documents.")
        return

    # Populate the dropdown with the titles from the dictionaries
    selected_document = st.selectbox(
        "Select a document for research:",
        [pdf['title'] for pdf in st.session_state['pdf_data']],
        help="Choose a document from the available list."
    )

    # Get the link for the selected document
    selected_document_url = next(
        (pdf['link'] for pdf in st.session_state['pdf_data'] if pdf['title'] == selected_document),
        None
    )

    # Extract the document name from the URL
    selected_document_name = extract_document_name(selected_document_url)

    # User research question
    st.subheader("Ask Your Research Question")
    user_question = st.text_input("Enter your question:")

    # Agent selection
    st.subheader("Select Agent")
    selected_agent = st.radio(
        "Choose an agent to answer your question:",
        ["RAG Agent", "Arxiv Agent", "Web Search Agent"]
    )

    # Conduct research when button is clicked
    if st.button("Submit Question"):
        if not selected_document_name or not user_question or not selected_agent:
            st.error("Please select a document, enter a question, and choose an agent to proceed.")
            return

        # Run the selected agent
        results = {}
        try:
            if selected_agent == "RAG Agent":
                results = RAGagent(selected_document_name, user_question).run()
            elif selected_agent == "Arxiv Agent":
                results = ArxivAgent(selected_document_name, user_question).run()
            elif selected_agent == "Web Search Agent":
                results = WebSearchAgent(selected_document_name, user_question).run()
        except Exception as e:
            st.error(f"Error: {e}")
            return

        # Display results
        st.subheader("Research Results")
        st.write(f"**Document**: {selected_document_name}")
        st.write(f"**Answer**: {results.get('answer', 'No answer generated.')}")
        st.write(f"**Details**: {results.get('details', 'No details provided.')}")
        
        # Append to chat history
        st.session_state['chat_history'].append((user_question, results.get('answer', 'No answer generated.')))

    # Display chat history
    st.subheader("Chat History")
    if st.session_state['chat_history']:
        for i, (question, answer) in enumerate(st.session_state['chat_history'], start=1):
            st.text(f"Q{i}: {question}")
            st.text(f"A: {answer}")

    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state['chat_history'] = []
        st.success("Chat history cleared.")
        st.rerun()

# Main Interface
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Research"])
    if page == "Research":
        if not st.session_state['pdf_data']:
            st.session_state['pdf_data'] = fetch_pdf_data_from_s3()
        research_interface()

# Load the main interface directly
main()