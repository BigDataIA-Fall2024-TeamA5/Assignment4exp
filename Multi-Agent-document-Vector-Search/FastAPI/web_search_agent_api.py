from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from jwtauth import get_current_user  # Import authentication dependency
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from web_search_agent import WebSearchAgent

# Create a router for WebSearchAgent API
router = APIRouter(prefix="/web_search", tags=["Web Search"])

# Pydantic model for the request body
class WebSearchRequest(BaseModel):
    selected_document: str
    user_query: str
    num_results: int = 10

@router.get("/search")
def web_search_get(query: str, num_results: int = 10, current_user: dict = Depends(get_current_user)):
    """GET endpoint to perform a web search."""
    try:
        agent = WebSearchAgent(selected_document="GET request", user_query=query, num_results=num_results)
        results = agent.search()
        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
def web_search_post(request: WebSearchRequest, current_user: dict = Depends(get_current_user)):
    """POST endpoint to perform a web search using a request body."""
    try:
        agent = WebSearchAgent(
            selected_document=request.selected_document, 
            user_query=request.user_query,
            num_results=request.num_results
        )
        results = agent.search()
        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
