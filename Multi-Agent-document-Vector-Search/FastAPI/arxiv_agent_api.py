from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from jwtauth import get_current_user  # Import authentication dependency
import sys
import os

# Add Streamlit folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Streamlit")))
from arxiv_agent import ArxivAgent

# Create a router for ArxivAgent API
router = APIRouter(prefix="/arxiv", tags=["Arxiv"])

# Pydantic model for the request body
class ArxivRequest(BaseModel):
    selected_document: str
    user_query: str
    num_results: int = 10

@router.get("/search")
def search_arxiv_get(query: str, num_results: int = 10, current_user: dict = Depends(get_current_user)):
    """GET endpoint to search Arxiv with a user query."""
    try:
        agent = ArxivAgent(selected_document="GET request", user_query=query)
        results = agent.search_arxiv(query, num_results=num_results)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
def search_arxiv_post(request: ArxivRequest, current_user: dict = Depends(get_current_user)):
    """POST endpoint to search Arxiv using a request body."""
    try:
        agent = ArxivAgent(
            selected_document=request.selected_document, 
            user_query=request.user_query
        )
        results = agent.search_arxiv(request.user_query, num_results=request.num_results)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
