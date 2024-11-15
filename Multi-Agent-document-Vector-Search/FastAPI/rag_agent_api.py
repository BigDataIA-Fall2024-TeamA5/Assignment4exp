from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from jwtauth import get_current_user  # Import authentication dependency
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag_agent import RAGAgent

# Create a router for RAGAgent API
router = APIRouter(prefix="/rag", tags=["RAG Agent"])

# Pydantic model for the request body
class RAGRequest(BaseModel):
    document_name: str
    query: str

@router.get("/process")
def rag_process_get(document_name: str, query: str, current_user: dict = Depends(get_current_user)):
    """GET endpoint to process a query using the RAG agent."""
    try:
        agent = RAGAgent(document_name=document_name, query=query)
        response = agent.run()
        return {"status": "success", "data": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process")
def rag_process_post(request: RAGRequest, current_user: dict = Depends(get_current_user)):
    """POST endpoint to process a query using the RAG agent."""
    try:
        agent = RAGAgent(document_name=request.document_name, query=request.query)
        response = agent.run()
        return {"status": "success", "data": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
