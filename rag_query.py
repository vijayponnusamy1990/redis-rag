"""
Retrieval-only RAG with Confluence + Redis (RediSearch vector index)
- USE_OLLAMA=true → embeddings via Ollama `nomic-embed-text`
- Otherwise, OpenAI or SentenceTransformers fallback
- Redis vector index already ingested
"""

import os
import sys
import logging
import numpy as np
import requests
from dotenv import load_dotenv
from typing import List
from redis.commands.search.query import Query

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --------------------------
# Check mandatory env vars
# --------------------------
for var in ["CONFLUENCE_BASE_URL", "CONFLUENCE_EMAIL", "CONFLUENCE_API_TOKEN", "CONFLUENCE_SPACE_KEY"]:
    if not os.getenv(var):
        logging.error("Environment variable %s is not set!", var)
        sys.exit(1)

# --------------------------
# Config from env
# --------------------------
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() in ("1", "true", "yes")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
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
            import openai
            openai.api_key = OPENAI_API_KEY
            self.openai = openai
        else:
            self.openai = None

        if not OPENAI_API_KEY and not self.use_ollama:
            from sentence_transformers import SentenceTransformer
            global _local_embedding_model
            if _local_embedding_model is None:
                logging.info("Loading local SentenceTransformer: all-MiniLM-L6-v2")
                _local_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.local = _local_embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.use_ollama:
            embeddings = []
            for t in texts:
                logging.debug("Requesting Ollama embedding for text: %s...", t[:50])
                resp = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": OLLAMA_EMBED_MODEL, "prompt": t},
                    timeout=120,
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            return embeddings
        elif self.openai:
            res = self.openai.Embedding.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
            return [r["embedding"] for r in res["data"]]
        else:
            logging.debug("Using local SentenceTransformer embeddings")
            return [list(v) for v in self.local.encode(texts)]

# --------------------------
# Redis client
# --------------------------
def get_redis_client():
    import redis
    r = redis.from_url(REDIS_URL)
    logging.debug("Connected to Redis at %s", REDIS_URL)
    return r

# --------------------------
# RAG retrieval
# --------------------------
def rag_query(query_text: str, emb: EmbeddingBackend, r, top_k: int = TOP_K, show_full_text=False):
    logging.info("Running RAG query: %s", query_text)
    
    # 1️⃣ Generate embedding
    try:
        emb_vec = emb.embed([query_text])[0]
        logging.debug("Query embedding length: %d", len(emb_vec))
    except Exception as e:
        logging.error("Failed to generate embedding: %s", e)
        return

    query_bytes = np.array(emb_vec, dtype=np.float32).tobytes()

    # 2️⃣ Search Redis
    try:
        q = (
            Query("*=>[KNN $k @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("title", "text", "page_url", "score")
            .paging(0, top_k)
            .dialect(2)
        )
        res = r.ft(REDIS_INDEX).search(q, query_params={"vec": query_bytes, "k": top_k})
        logging.debug("Redis search completed")
    except Exception as e:
        logging.error("Redis search failed: %s", e)
        return

    # 3️⃣ Print table
    if res and res.docs:
        print(f"\nTop {len(res.docs)} results for query: '{query_text}'\n")
        print("{:<4} {:<40} {:<60} {:<8}".format("No", "Title", "URL", "Score"))
        print("-"*120)
        for i, doc in enumerate(res.docs, 1):
            try:
                score = float(getattr(doc, "score", 0))
            except:
                score = 0.0
            print("{:<4} {:<40} {:<60} {:<8.4f}".format(i, doc.title, doc.page_url, score))
            if show_full_text:
                print("\n--- Document snippet ---")
                print(doc.text[:1000])
                print("------------------------\n")
    else:
        logging.warning("No results found for query.")

# --------------------------
# Main
# --------------------------
def main():
    query = input("Enter your query: ").strip()
    r = get_redis_client()
    emb = EmbeddingBackend()
    rag_query(query, emb, r, TOP_K, show_full_text=True)

if __name__ == "__main__":
    main()
