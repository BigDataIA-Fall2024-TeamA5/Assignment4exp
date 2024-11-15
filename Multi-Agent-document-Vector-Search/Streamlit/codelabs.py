import os
import markdown
from datetime import datetime
from dotenv import load_dotenv
import subprocess
import streamlit as st

# Load environment variables
load_dotenv()

# Directory to save reports
CODELABS_DIR = "codelabs"
os.makedirs(CODELABS_DIR, exist_ok=True)

# Function to generate markdown from chat history
def generate_markdown_for_codelabs(chat_history):
    """Create a markdown string formatted for Codelabs from chat history."""
    # Ensure valid metadata
    codelabs_content = """summary: Codelabs Chat History
id: chat-history-codelab
categories: Machine Learning
tags: AI, Q&A
status: Draft
authors: Your Name
feedback link: https://example.com/feedback
---
"""

    # Add title and introduction
    codelabs_content += "# Codelabs Chat History\n\n"
    codelabs_content += "## Introduction\n"
    codelabs_content += "This Codelabs file documents the Q&A interactions.\n\n"

    # Add questions and answers as steps
    for idx, (question, answer) in enumerate(chat_history, 1):
        codelabs_content += f"### Step {idx}: {question}\n\n"  # Add question
        codelabs_content += f"- **Answer:** {answer}\n\n"  # Add answer

    return codelabs_content



# Function to save Codelabs markdown file
def save_codelabs_file(markdown_content, output_path):
    """Save the generated markdown content as a .md file for Codelabs."""
    try:
        with open(output_path, "w") as file:
            file.write(markdown_content)
        return True
    except Exception as e:
        print(f"Error saving Codelabs file: {e}")
        return False


def run_claat(output_file):
    """Run claat on the generated markdown file."""
    try:
        claat_path = "/Users/aniketpatole/Documents/GitHub/New/Projects/BigData/Assignment4exp/Multi-Agent-document-Vector-Search/Streamlit/claat-darwin-amd64"
        result = subprocess.run(
            [claat_path, "export", output_file],
            check=True,
            capture_output=True,
            text=True
        )
        st.success("Codelabs document generated successfully!")
        codelabs_output_dir = os.path.join(CODELABS_DIR, "output")
        return codelabs_output_dir
    except subprocess.CalledProcessError as e:
        st.error(f"Error running claat: {e.stderr}")
        return None


# Function to handle Codelabs generation
def generate_codelabs(chat_history):
    """Handle Codelabs file generation from chat history."""
    # Generate Codelabs markdown content
    codelabs_content = generate_markdown_for_codelabs(chat_history)

    # Create output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(CODELABS_DIR, f"chat_history_{timestamp}.md")

    # Save the Codelabs markdown file
    if save_codelabs_file(codelabs_content, output_file):
        st.success(f"Codelabs markdown file generated successfully! Saved at: {output_file}")

        # Run claat to generate the Codelabs document
        codelabs_output = run_claat(output_file)
        if codelabs_output:
            st.success(f"Codelabs document exported to: {codelabs_output}")
            return codelabs_output

    else:
        st.error("Failed to generate Codelabs markdown file.")
        return None
