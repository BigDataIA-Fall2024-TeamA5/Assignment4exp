from langgraph.graph import Node
from typing import List, Dict


class DocumentAgent:
    def __init__(self, available_documents: List[Dict]):
        """
        Initialize the DocumentAgent with the list of available documents.

        Args:
            available_documents (List[Dict]): List of dictionaries with document metadata.
                                              Example: [{"name": "Doc1", "link": "s3://path/to/doc1.pdf"}, ...]
        """
        self.available_documents = available_documents

    def list_documents(self) -> List[str]:
        """
        List all available document names for selection.

        Returns:
            List[str]: A list of document names.
        """
        return [doc["name"] for doc in self.available_documents]

    def select_document(self, document_name: str) -> Dict:
        """
        Select a document based on the document name.

        Args:
            document_name (str): The name of the document to select.

        Returns:
            Dict: The selected document's metadata (e.g., name, link).
        """
        for document in self.available_documents:
            if document["name"] == document_name:
                return document
        raise ValueError(f"Document '{document_name}' not found in the available documents.")

    def run(self, inputs: Dict) -> Dict:
        """
        Run the DocumentAgent node in the Langraph workflow.

        Args:
            inputs (Dict): Input payload containing a "document_name" key.

        Returns:
            Dict: Metadata of the selected document.
        """
        document_name = inputs.get("document_name")
        if not document_name:
            return {"error": "No document name provided"}

        try:
            selected_document = self.select_document(document_name)
            return {"selected_document": selected_document}
        except ValueError as e:
            return {"error": str(e)}