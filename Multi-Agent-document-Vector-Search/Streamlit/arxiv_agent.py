import arxiv
import nltk
from nltk.corpus import wordnet as wn
import spacy

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

class ArxivAgent:
    def __init__(self, selected_document: str, user_query: str):
        self.selected_document = selected_document
        self.user_query = user_query

    def expand_query_with_synonyms(self, query: str) -> str:
        """Expand the query by adding synonyms using NLTK and SpaCy."""
        doc = nlp(query)
        expanded_terms = set()
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.text not in expanded_terms:
                expanded_terms.add(token.text)
                for syn in wn.synsets(token.text):
                    for lemma in syn.lemmas()[:3]:  # Limit to 3 synonyms
                        expanded_terms.add(lemma.name().replace("_", " "))
        expanded_query = query + " " + " ".join(expanded_terms)
        return expanded_query

    def search_arxiv(self, query: str, num_results: int = 10) -> list:
        """Search Arxiv for relevant papers."""
        expanded_query = self.expand_query_with_synonyms(query)
        
        search = arxiv.Search(
            query=expanded_query,
            max_results=num_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        for result in search.results():
            paper_info = {
                "title": result.title,
                "summary": result.summary,
                "pdf_url": result.pdf_url
            }
            papers.append(paper_info)
        return papers

    def run(self) -> dict:
        """Execute the agent to search Arxiv."""
        print(f"Running ArxivAgent for document: {self.selected_document} and query: {self.user_query}")
        results = self.search_arxiv(self.user_query)
        if not results:
            return {"answer": "No relevant papers found on Arxiv.", "details": ""}
        
        # Format results for output
        formatted_results = "\n".join(
            [f"Title: {paper['title']}\nSummary: {paper['summary']}\nPDF: {paper['pdf_url']}\n" for paper in results]
        )
        return {"answer": "Arxiv search completed. See details below.", "details": formatted_results}
