import os
import markdown
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Directory to save reports
CODELABS_DIR = "codelabs"
os.makedirs(CODELABS_DIR, exist_ok=True)

# Function to generate markdown from chat history
def generate_markdown_for_codelabs(chat_history):
    """Create a markdown string formatted for Codelabs from chat history."""
    codelabs_content = "# Codelabs Chat History\n\n"
    codelabs_content += "## Introduction\n"
    codelabs_content += "This Codelabs file documents the Q&A interactions.\n\n"

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
        st.success(f"Codelabs file generated successfully! Saved at: {output_file}")
        return output_file
    else:
        st.error("Failed to generate Codelabs file.")
        return None
