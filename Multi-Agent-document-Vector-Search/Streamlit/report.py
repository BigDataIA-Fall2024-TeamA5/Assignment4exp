import os
import pdfkit
import markdown
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Directory to save reports
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Function to generate markdown from chat history
def generate_markdown(chat_history):
    """Create a markdown string from chat history."""
    markdown_content = "# Chat History Report\n\n"
    for idx, (question, answer) in enumerate(chat_history, 1):
        markdown_content += f"## Q{idx}: {question}\n\n"  # Add question
        markdown_content += f"**Answer:** {answer}\n\n"  # Add answer
    return markdown_content

# Function to convert markdown to PDF
def generate_pdf_from_markdown(markdown_content, output_path):
    """Convert markdown content to PDF using pdfkit."""
    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content)

    # Save the PDF
    try:
        pdfkit.from_string(html_content, output_path)
        return True
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

# Streamlit button and report generation
def generate_report(chat_history):
    """Handle report generation from chat history."""
    # Generate markdown content
    markdown_content = generate_markdown(chat_history)

    # Create output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(REPORT_DIR, f"chat_history_{timestamp}.pdf")

    # Generate PDF
    if generate_pdf_from_markdown(markdown_content, output_file):
        st.success(f"Report generated successfully! Saved at: {output_file}")
        return output_file
    else:
        st.error("Failed to generate report.")
        return None

