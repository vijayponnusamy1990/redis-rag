"""
RAG query from Redis vector index

- USE_OLLAMA=true → embeddings via Ollama `nomic-embed-text`
- Otherwise, OpenAI or SentenceTransformers fallback
- Debug logging added
"""

import os
import logging
import requests
import numpy as np
from typing import List
from dotenv import load_dotenv
import openai
from sentence_transformers import SentenceTransformer

import redis
from redis.commands.search.query import Query

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

load_dotenv()

# Config
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() in ("1", "true", "yes")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_NAMESPACE = os.getenv("REDIS_NAMESPACE", "confluence:rdb")
REDIS_INDEX = os.getenv("REDIS_VECTOR_INDEX", "confluence_idx")
TOP_K = int(os.getenv("TOP_K", "5"))

_local_embedding_model = None

# --------------------------
# Embeddings backend
# --------------------------
class EmbeddingBackend:
    def __init__(self):
        self.use_ollama = USE_OLLAMA
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        if not OPENAI_API_KEY and not self.use_ollama:
            global _local_embedding_model
            if _local_embedding_model is None:
                logging.info("Loading local SentenceTransformer: all-MiniLM-L6-v2")
                _local_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.local = _local_embedding_model

    def _embed_ollama(self, text: str) -> List[float]:
        logging.debug("Requesting Ollama embedding for text: %.50s...", text)
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.use_ollama:
            embeddings = []
            for t in texts:
                embeddings.append(self._embed_ollama(t))
            return embeddings
        elif OPENAI_API_KEY:
            logging.debug(f"Requesting OpenAI embeddings for {len(texts)} texts")
            res = openai.Embedding.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
            return [r["embedding"] for r in res["data"]]
        else:
            logging.debug(f"Using local SentenceTransformer for {len(texts)} texts")
            return [list(v) for v in self.local.encode(texts)]

# --------------------------
# Redis client
# --------------------------
def get_redis_client():
    r = redis.from_url(REDIS_URL)
    logging.debug(f"Connected to Redis at {REDIS_URL}")
    return r

# --------------------------
# RAG query
# --------------------------
def rag_query(query_text: str, emb: EmbeddingBackend, r: redis.Redis, top_k: int = 5, snippet_len: int = 200):
    """
    Perform a RAG query: embed the query, search Redis vector index, and log top-k results.
    Returns a list of result dicts: [{'title': ..., 'url': ..., 'score': ..., 'snippet': ...}, ...]
    """
    logging.info("Running RAG query: %s", query_text)

    try:
        emb_vec = emb.embed([query_text])[0]
        logging.debug("Query embedding length: %d", len(emb_vec))
    except Exception as e:
        logging.error("Failed to get embedding: %s", e)
        return []

    query_bytes = np.array(emb_vec, dtype=np.float32).tobytes()

    try:
        # Redis Stack vector query
        q = (
            Query("*")
            .return_fields("title", "text", "page_url")
            .paging(0, top_k)
            .dialect(2)  # Required for vector search params
        )
        res = r.ft(REDIS_INDEX).search(q, query_params={"vec": query_bytes})
        logging.debug("Redis search completed")
    except Exception as e:
        logging.error("Redis search failed: %s", e)
        return []

    results = []
    if res and res.docs:
        logging.info("Found %d results", len(res.docs))
        for doc in res.docs:
            # Ensure score is a float for logging
            try:
                score = float(getattr(doc, "score", 0.0))
            except ValueError:
                score = 0.0

            snippet = getattr(doc, "text", "")[:snippet_len].replace("\n", " ").strip()
            logging.info("Title: %s | URL: %s | Score: %.4f", doc.title, doc.page_url, score)
            logging.debug("Text snippet: %s", snippet)

            results.append({
                "title": doc.title,
                "url": doc.page_url,
                "score": score,
                "snippet": snippet,
            })
    else:
        logging.warning("No results found for query.")

    return results



# --------------------------
# Main
# --------------------------
def main():
    emb = EmbeddingBackend()
    r = get_redis_client()

    results = rag_query("What is Indian history after 1900?", emb, r)
    for r in results:
        print(r["title"], r["score"], r["url"])

if __name__ == "__main__":
    main()
