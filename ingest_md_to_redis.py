"""
Ingest Confluence Cloud pages, chunk, embed, and store in Redis (RediSearch vector index).
"""

import os
import time
import html
import requests
import logging
from dotenv import load_dotenv
from urllib.parse import urljoin
from typing import List

import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# embedding backends
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import tiktoken

# -------------------------------
# Logging setup
# -------------------------------
logger = logging.getLogger("confluence_ingest")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# -------------------------------
# Load environment
# -------------------------------
load_dotenv()

BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
EMAIL = os.getenv("CONFLUENCE_EMAIL")
API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_NAMESPACE = os.getenv("REDIS_NAMESPACE", "confluence:rdb")
REDIS_INDEX = os.getenv("REDIS_VECTOR_INDEX", "confluence_idx")

USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() in ("1", "true", "yes")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "0"))

_local_embedding_model = None

# -------------------------------
# Helper functions
# -------------------------------
def get_sentence_token_count(text: str, model_name: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


def text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    logger.debug("Splitting text into chunks (chunk_size=%s, overlap=%s)", chunk_size, overlap)
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    chunks = []
    cur = []
    cur_tokens = 0
    for s in sentences:
        s_tokens = get_sentence_token_count(s)
        if cur_tokens + s_tokens > chunk_size and cur:
            chunks.append(". ".join(cur).strip())
            logger.debug("Created chunk with %d tokens", cur_tokens)
            # keep overlap
            keep = []
            keep_tokens = 0
            while cur and keep_tokens < overlap:
                last = cur.pop()
                keep.insert(0, last)
                keep_tokens += get_sentence_token_count(last)
            cur = keep
            cur_tokens = sum(get_sentence_token_count(x) for x in cur)
        cur.append(s)
        cur_tokens += s_tokens
    if cur:
        chunks.append(". ".join(cur).strip())
        logger.debug("Created final chunk with %d tokens", cur_tokens)
    return chunks


def html_to_text(html_content: str) -> str:
    import re
    text = re.sub(r"<br\s*/?>", "\n", html_content)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text

# -------------------------------
# Confluence Client
# -------------------------------
class ConfluenceClient:
    def __init__(self, base_url, email, token):
        self.base_url = base_url.rstrip("/") + "/"
        self.session = requests.Session()
        self.session.auth = (email, token)
        self.session.headers.update({"Accept": "application/json"})

    def fetch_pages(self, space_key=None, limit=50, start=0):
        params = {"limit": limit, "start": start, "expand": "body.storage,version"}
        if space_key:
            params["spaceKey"] = space_key
        url = urljoin(self.base_url, "rest/api/content")
        logger.debug("Fetching pages from %s (start=%d, limit=%d)", url, start, limit)
        r = self.session.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def iter_pages(self, space_key=None, max_pages=0):
        start = 0
        page_count = 0
        while True:
            data = self.fetch_pages(space_key=space_key, limit=50, start=start)
            results = data.get("results", [])
            if not results:
                break
            for p in results:
                page_count += 1
                logger.debug("Fetched page %s: %s", p.get("id"), p.get("title"))
                yield p
                if max_pages and page_count >= max_pages:
                    return
            start += len(results)

# -------------------------------
# Embedding Backend
# -------------------------------
class EmbeddingBackend:
    def __init__(self):
        self.use_ollama = USE_OLLAMA
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY

        if not OPENAI_API_KEY and not self.use_ollama:
            global _local_embedding_model
            if _local_embedding_model is None:
                _local_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.debug("Loaded local SentenceTransformer model")
            self.local = _local_embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        logger.debug("Embedding %d texts", len(texts))
        if self.use_ollama:
            embeddings = []
            for t in texts:
                payload = {"model": OLLAMA_MODEL, "input": t}
                logger.debug("Calling Ollama embed model=%s", OLLAMA_MODEL)
                resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                embeddings.append(data["embedding"])
            return embeddings
        elif OPENAI_API_KEY:
            res = openai.Embedding.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
            return [r['embedding'] for r in res['data']]
        else:
            return [list(v) for v in self.local.encode(texts)]

# -------------------------------
# Redis helpers
# -------------------------------
def get_redis_client():
    logger.debug("Connecting to Redis: %s", REDIS_URL)
    return redis.from_url(REDIS_URL)


def create_vector_index(r: redis.Redis, dim: int, prefix: str):
    try:
        logger.debug("Dropping existing index %s if exists", REDIS_INDEX)
        r.ft(REDIS_INDEX).dropindex(delete_documents=False)
    except Exception:
        logger.debug("No existing index to drop")

    try:
        logger.debug("Creating RediSearch vector index %s with dim=%d", REDIS_INDEX, dim)
        schema = (
            TextField("title", weight=5.0),
            TextField("page_url"),
            TextField("page_id"),
            VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": 1000,
            }),
        )
        idx_def = IndexDefinition(prefix=[prefix + ":"], index_type=IndexType.HASH)
        r.ft(REDIS_INDEX).create_index(schema, definition=idx_def)
        logger.info("✅ Recreated RediSearch vector index: %s", REDIS_INDEX)
    except Exception as e:
        logger.warning("Index creation warning: %s", e)


def store_chunk(r: redis.Redis, prefix: str, doc_id: str, title: str, page_url: str, embedding: List[float], text: str):
    vec = np.array(embedding, dtype=np.float32).tobytes()
    key = f"{prefix}:{doc_id}"
    logger.debug("Storing chunk key=%s title=%s", key, title)
    r.hset(key, mapping={
        "title": title,
        "page_url": page_url,
        "page_id": doc_id,
        "text": text,
        "embedding": vec
    })


# -------------------------------
# Main ingestion
# -------------------------------
def main():
    logger.info("Starting Confluence ingestion...")
    client = ConfluenceClient(BASE_URL, EMAIL, API_TOKEN)
    emb = EmbeddingBackend()
    r = get_redis_client()

    pages = list(client.iter_pages(space_key=SPACE_KEY, max_pages=MAX_PAGES))
    logger.info("Fetched %d pages", len(pages))

    all_chunks, metadata = [], []
    for page in pages:
        page_id = page.get("id")
        title = page.get("title")
        storage = page.get("body", {}).get("storage", {}).get("value", "")
        text = html_to_text(storage)
        chunks = text_to_chunks(text)
        logger.debug("Page %s produced %d chunks", page_id, len(chunks))
        for i, ch in enumerate(chunks):
            doc_id = f"{page_id}:{i}"
            all_chunks.append(ch)
            metadata.append({
                "doc_id": doc_id,
                "title": title,
                "page_url": urljoin(BASE_URL, f"/pages/viewpage.action?pageId={page_id}")
            })

    if not all_chunks:
        logger.warning("No content to index. Exiting.")
        return

    BATCH = 16
    embeddings = []
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        es = emb.embed(batch)
        embeddings.extend(es)
        logger.debug("Embedded batch %d-%d", i, i + len(batch))
        time.sleep(0.05)

    dim = len(embeddings[0])
    create_vector_index(r, dim, REDIS_NAMESPACE)

    for md, emb_vec, text in zip(metadata, embeddings, all_chunks):
        store_chunk(r, REDIS_NAMESPACE, md['doc_id'], md['title'], md['page_url'], emb_vec, text)

    logger.info("✅ Stored %d chunks into Redis namespace %s", len(all_chunks), REDIS_NAMESPACE)


if __name__ == '__main__':
    main()
