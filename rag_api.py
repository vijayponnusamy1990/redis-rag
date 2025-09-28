from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from rag_query_llm_cache import rag_query, get_redis_client, TOP_K

# Import main from ingest_confluence
from ingest_confluence import main as ingest_confluence_main
from ingest_pdf import main as ingest_pdf_main

app = FastAPI(title="RAG API")

class RAGRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

class RAGResponse(BaseModel):
    answer: str

@app.post("/rag", response_model=RAGResponse)
def rag_endpoint(req: RAGRequest):
    r = get_redis_client()
    answer = rag_query(req.query, r, TOP_K)
    return RAGResponse(answer=answer)

@app.post("/ingest/confluence")
def ingest_confluence_endpoint():
    """
    Triggers the Confluence ingestion process.
    """
    ingest_confluence_main()
    return {"status": "Ingestion triggered"}

@app.post("/ingest/pdf")
def ingest_pdf_endpoint():
    """
    Triggers the PDF ingestion process.
    """
    ingest_pdf_main()
    return {"status": "PDF ingestion triggered"}
