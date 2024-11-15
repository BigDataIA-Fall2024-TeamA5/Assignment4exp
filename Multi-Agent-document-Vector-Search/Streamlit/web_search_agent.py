import os
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WebSearchAgent:
    def __init__(self, selected_document, user_query, num_results=10):
        """
        Initialize the WebSearchAgent with Tavily client, query details, and number of results.

        Args:
            selected_document (str): The document selected for the research.
            user_query (str): The search query provided by the user.
            num_results (int): Number of search results to retrieve. Default is 10.
        """
        self.selected_document = selected_document
        self.user_query = user_query
        self.num_results = num_results

        self.api_key = os.getenv("TAVILY_API_KEY")

        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables. Please add it to your .env file.")

        self.client = TavilyClient(api_key=self.api_key)

    def search(self):
        """
        Perform a web search using the Tavily client.

        Returns:
            list: A list of search results, each containing a title, URL, and snippet.
        """
        try:
            results = self.client.search(query=self.user_query, limit=self.num_results)
            return [
                {
                    "title": result.get("title", "No title available"),
                    "url": result.get("url", "No URL available"),
                    "snippet": result.get("snippet", "No snippet available")
                }
                for result in results.get("results", [])
            ]
        except Exception as e:
            return {"error": f"Error during Tavily search: {str(e)}"}

    def run(self):
        """
        Execute the web search agent logic.

        Returns:
            dict: A dictionary containing the search results or an error message.
        """
        print(f"Running WebSearchAgent for document: {self.selected_document} and query: {self.user_query}")
        search_results = self.search()
        if isinstance(search_results, dict) and "error" in search_results:
            return {"answer": "Error occurred during web search.", "details": search_results["error"]}

        formatted_results = "\n".join(
            [f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}" for result in search_results]
        )
        return {"answer": "Web search completed. See details below.", "details": formatted_results}


# Example usage
if __name__ == "__main__":
    agent = WebSearchAgent(
        selected_document="Example Document.pdf",
        user_query="How does AI impact investments?",
        num_results=5
    )

    response = agent.run()
    print("Answer:", response["answer"])
    print("Details:", response["details"])