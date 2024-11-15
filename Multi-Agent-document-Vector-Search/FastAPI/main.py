from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from jwtauth import router as auth_router
from arxiv_agent_api import router as arxiv_router
from web_search_agent_api import router as web_search_router
from rag_agent_api import router as rag_router
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()

app = FastAPI()

# Include the JWT auth router
app.include_router(auth_router, prefix="/auth")

# Include the ArxivAgent API router
app.include_router(arxiv_router)

# Include the WebSearchAgent API router
app.include_router(web_search_router)

# Include the RAGAgent API router
app.include_router(rag_router)

# Define the OAuth2 scheme 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI JWT Authentication Application!"}
