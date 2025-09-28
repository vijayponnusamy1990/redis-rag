import os
import sys
import json
import logging
import requests
import hashlib
import numpy as np
from dotenv import load_dotenv
from typing import List
from redis.commands.search.query import Query
from abc import ABC, abstractmethod
from openai import AzureOpenAI

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --------------------------
# Config
# --------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "azure"

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_INDEX = os.getenv("REDIS_VECTOR_INDEX", "confluence_idx")
REDIS_CACHE_KEY = os.getenv("REDIS_CACHE_KEY", "llm_cache")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # default 1 hour
TOP_K = int(os.getenv("TOP_K", "5"))


# --------------------------
# Redis client
# --------------------------
def get_redis_client():
    import redis
    r = redis.from_url(REDIS_URL)
    logging.debug("Connected to Redis at %s", REDIS_URL)
    return r


# ================================================================
# LLM Providers (Factory pattern + embed)
# ================================================================
class LLMProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass

    @abstractmethod
    def _call_llm(self, query: str, context: str) -> str:
        """Actual LLM call, to be implemented by subclasses"""
        pass

    def generate(self, query: str, context: str, r) -> str:
        """Wrapper: check cache before hitting LLM"""
        key_raw = f"{query}:{context}"
        key_hash = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

        cached = r.hget(REDIS_CACHE_KEY, key_hash)
        if cached:
            logging.info("Cache hit for query")
            return cached.decode("utf-8")

        logging.info("Cache miss → calling LLM")
        answer = self._call_llm(query, context)

        if answer:
            r.hset(REDIS_CACHE_KEY, key_hash, answer)
            r.expire(REDIS_CACHE_KEY, CACHE_TTL)

        return answer


class OllamaProvider(LLMProvider):
    def embed(self, texts: List[str]) -> List[List[float]]:
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

    def _call_llm(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful assistant. 
Use ONLY the following context to answer the question. 
If the context is insufficient, say so and provide links to the sources. 
Do NOT hallucinate.

Context:
{context}

Question: {query}
Answer:"""

        logging.debug("Sending prompt to Ollama LLM (model=%s)", OLLAMA_LLM_MODEL)

        try:
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
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        answer += data.get("response", "")
                except Exception:
                    continue
            return answer.strip()
        except Exception as e:
            logging.error(f"Ollama call failed: {e}")
            return "Error: Ollama request failed."


class AzureOpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            logging.debug("Requesting Azure OpenAI embeddings")
            response = self.client.embeddings.create(
                model=AZURE_OPENAI_EMBED_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logging.error(f"Azure embedding failed: {e}")
            return []

    def _call_llm(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful assistant. 
Use ONLY the following context to answer the question. 
If the context is insufficient, say so and provide links to the sources. 
Do NOT hallucinate.

Context:
{context}

Question: {query}
Answer:"""

        logging.debug("Sending prompt to Azure OpenAI LLM (deployment=%s)", AZURE_OPENAI_DEPLOYMENT)

        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a RAG assistant. Only use provided context."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Azure OpenAI call failed: {e}")
            return "Error: Azure OpenAI request failed."


class LLMFactory:
    @staticmethod
    def get_provider() -> LLMProvider:
        if LLM_PROVIDER == "ollama":
            return OllamaProvider()
        elif LLM_PROVIDER == "azure":
            return AzureOpenAIProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


# --------------------------
# RAG retrieval
# --------------------------
def rag_query(query_text: str, r, top_k: int = TOP_K):
    logging.info("Running RAG query: %s", query_text)

    llm = LLMFactory.get_provider()

    # 1️⃣ Generate embedding
    emb_vec = llm.embed([query_text])[0]
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

    if not docs:
        return "No relevant documents found."

    # 4️⃣ Build context string
    context_str = "\n\n".join(
        f"Title: {doc.title}\nURL: {doc.page_url}\nContent:\n{doc.text[:1500]}"
        for doc in docs
    )

    # 5️⃣ Ask chosen LLM (with cache)
    enriched = llm.generate(query_text, context_str, r)

    return enriched


# --------------------------
# Main
# --------------------------
def main():
    query = input("Enter your query: ").strip()
    r = get_redis_client()
    enriched_answer = rag_query(query, r, TOP_K)

    print("\nLLM Enriched Answer:\n", enriched_answer)


if __name__ == "__main__":
    main()
