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
# Config
# --------------------------
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() in ("1", "true", "yes")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_INDEX = os.getenv("REDIS_VECTOR_INDEX", "confluence_idx")
TOP_K = int(os.getenv("TOP_K", "5"))

_local_embedding_model = None


# --------------------------
# Embeddings backend
# --------------------------
class EmbeddingBackend:
    def __init__(self):
        if not USE_OLLAMA:
            from sentence_transformers import SentenceTransformer
            global _local_embedding_model
            if _local_embedding_model is None:
                logging.info("Loading local SentenceTransformer: all-MiniLM-L6-v2")
                _local_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.local = _local_embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if USE_OLLAMA:
            embeddings = []
            for t in texts:
                logging.debug("Requesting Ollama embedding for: %s...", t[:50])
                resp = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": OLLAMA_EMBED_MODEL, "prompt": t},
                    timeout=120,
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            return embeddings
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
# LLM call
# --------------------------
def ask_ollama_llm(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. 
Use ONLY the following context to answer the question. 
If the context is insufficient, say so and provide links to the sources. 
Do NOT hallucinate.

Context:
{context}

Question: {question}
Answer:"""

    logging.debug("Sending prompt to Ollama LLM (model=%s)", OLLAMA_LLM_MODEL)

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_LLM_MODEL, "prompt": prompt},
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    answer = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = line.decode("utf-8")
            if '"response"' in data:
                import json
                parsed = json.loads(data)
                answer += parsed.get("response", "")
        except Exception:
            continue
    return answer.strip()


# --------------------------
# RAG retrieval
# --------------------------
def rag_query(query_text: str, emb: EmbeddingBackend, r, top_k: int = TOP_K):
    logging.info("Running RAG query: %s", query_text)

    # 1️⃣ Generate embedding
    emb_vec = emb.embed([query_text])[0]
    query_bytes = np.array(emb_vec, dtype=np.float32).tobytes()

    # 2️⃣ Search Redis
    q = (
        Query("*=>[KNN $k @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("title", "text", "page_url", "score")
        .paging(0, top_k)
        .dialect(2)
    )
    res = r.ft(REDIS_INDEX).search(q, query_params={"vec": query_bytes, "k": top_k})
    logging.debug("Redis search completed")

    # 3️⃣ Collect results
    docs = []
    if res and res.docs:
        print(f"\nTop {len(res.docs)} results for query: '{query_text}'\n")
        print("{:<4} {:<40} {:<60} {:<8}".format("No", "Title", "URL", "Score"))
        print("-" * 120)
        for i, doc in enumerate(res.docs, 1):
            score = float(getattr(doc, "score", 0))
            print("{:<4} {:<40} {:<60} {:<8.4f}".format(i, doc.title, doc.page_url, score))
            docs.append(doc)
    else:
        logging.warning("No results found for query.")

    # 4️⃣ Build enriched context
    if not docs:
        return "No relevant documents found."

    context_str = "\n\n".join(
        f"Title: {doc.title}\nURL: {doc.page_url}\nContent:\n{doc.text[:1500]}"
        for doc in docs
    )

    # 5️⃣ Ask LLM
    enriched = ask_ollama_llm(query_text, context_str)

    return enriched


# --------------------------
# Main
# --------------------------
def main():
    query = input("Enter your query: ").strip()
    r = get_redis_client()
    emb = EmbeddingBackend()
    enriched_answer = rag_query(query, emb, r, TOP_K)

    print("\nLLM Enriched Answer:\n", enriched_answer)


if __name__ == "__main__":
    main()
