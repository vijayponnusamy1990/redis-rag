"""
Ingest Confluence Cloud pages, chunk, embed, and store in Redis (RediSearch vector index).

- If USE_OLLAMA=true → embeddings are created via Ollama `nomic-embed-text`.
- Otherwise, OpenAI or sentence-transformers fallback is used.
- Redis vector index is dropped and recreated on each run.
- Debug logging added.
"""

import os
import time
import html
import requests
import logging
from dotenv import load_dotenv
from urllib.parse import urljoin
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# embedding backends
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import tiktoken

# --- Setup ---
load_dotenv()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Config
BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
EMAIL = os.getenv("CONFLUENCE_EMAIL")
API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_NAMESPACE = os.getenv("REDIS_NAMESPACE", "confluence:rdb")
REDIS_INDEX = os.getenv("REDIS_VECTOR_INDEX", "confluence_idx")

USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() in ("1", "true", "yes")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "0"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

_local_embedding_model = None


def get_sentence_token_count(text: str, model_name: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


def text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    chunks, cur, cur_tokens = [], [], 0
    for s in sentences:
        s_tokens = get_sentence_token_count(s)
        if cur_tokens + s_tokens > chunk_size and cur:
            chunks.append(". ".join(cur).strip())
            logging.debug(f"Created chunk: {chunks[-1][:60]}...")
            keep, keep_tokens = [], 0
            while cur and keep_tokens < overlap:
                last = cur.pop()
                keep.insert(0, last)
                keep_tokens += get_sentence_token_count(last)
            cur, cur_tokens = keep, sum(get_sentence_token_count(x) for x in keep)
        cur.append(s)
        cur_tokens += s_tokens
    if cur:
        chunks.append(". ".join(cur).strip())
        logging.debug(f"Created chunk: {chunks[-1][:60]}...")
    return chunks


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
        logging.debug(f"Fetching pages from {url} with params {params}")
        r = self.session.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        logging.debug(f"Fetched {len(data.get('results', []))} pages")
        return data

    def iter_pages(self, space_key=None, max_pages=0):
        start, page_count = 0, 0
        while True:
            data = self.fetch_pages(space_key=space_key, limit=50, start=start)
            results = data.get("results", [])
            if not results:
                break
            for p in results:
                page_count += 1
                logging.debug(f"Yielding page {p.get('id')} - {p.get('title')}")
                yield p
                if max_pages and page_count >= max_pages:
                    return
            start += len(results)


def html_to_text(html_content: str) -> str:
    import re
    text = re.sub(r"<br\s*/?>", "\n", html_content)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)


class EmbeddingBackend:
    def __init__(self):
        self.use_ollama = USE_OLLAMA
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY

        if not OPENAI_API_KEY and not self.use_ollama:
            global _local_embedding_model
            if _local_embedding_model is None:
                _local_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.local = _local_embedding_model

    def _embed_ollama(self, text: str) -> List[float]:
        logging.debug(f"Requesting Ollama embedding for text: {text[:60]}...")
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.use_ollama:
            embeddings = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(self._embed_ollama, t): t for t in texts}
                for future in as_completed(futures):
                    try:
                        emb_vec = future.result()
                        embeddings.append(emb_vec)
                        logging.debug(f"Got embedding of length {len(emb_vec)}")
                    except Exception as e:
                        logging.warning(f"Failed to embed: {e}")
                        embeddings.append([0.0] * 768)
            return embeddings
        elif OPENAI_API_KEY:
            logging.debug(f"Requesting OpenAI embeddings for batch of {len(texts)} texts")
            res = openai.Embedding.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
            return [r["embedding"] for r in res["data"]]
        else:
            logging.debug(f"Using local SentenceTransformer for {len(texts)} texts")
            return [list(v) for v in self.local.encode(texts)]


def get_redis_client():
    r = redis.from_url(REDIS_URL)
    logging.debug(f"Connected to Redis at {REDIS_URL}")
    return r


def recreate_vector_index(r: redis.Redis, dim: int, prefix: str):
    try:
        r.ft(REDIS_INDEX).dropindex(delete_documents=True)
        logging.debug(f"Dropped existing index {REDIS_INDEX}")
    except Exception:
        logging.debug(f"No existing index to drop: {REDIS_INDEX}")

    schema = (
        TextField("title", weight=5.0),
        TextField("page_url"),
        TextField("page_id"),
        TextField("text"),
        VectorField("embedding", "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": dim,
            "DISTANCE_METRIC": "COSINE",
            "INITIAL_CAP": 1000,
        }),
    )
    idx_def = IndexDefinition(prefix=[prefix + ":"], index_type=IndexType.HASH)
    r.ft(REDIS_INDEX).create_index(schema, definition=idx_def)
    logging.debug(f"Recreated RediSearch vector index: {REDIS_INDEX}")


def store_chunk(r: redis.Redis, prefix: str, doc_id: str, title: str, page_url: str, embedding: List[float], text: str):
    vec = np.array(embedding, dtype=np.float32).tobytes()
    key = f"{prefix}:{doc_id}"
    r.hset(key, mapping={
        "title": title,
        "page_url": page_url,
        "page_id": doc_id,
        "text": text,
        "embedding": vec,
    })
    logging.debug(f"Stored chunk {doc_id} with {len(embedding)}-dim embedding")


def main():
    logging.info("Starting Confluence ingestion...")
    client = ConfluenceClient(BASE_URL, EMAIL, API_TOKEN)
    emb = EmbeddingBackend()
    r = get_redis_client()

    pages = list(client.iter_pages(space_key=SPACE_KEY, max_pages=MAX_PAGES))
    logging.info(f"Fetched {len(pages)} pages")

    all_chunks, metadata = [], []
    for page in pages:
        page_id = page.get("id")
        title = page.get("title")
        storage = page.get("body", {}).get("storage", {}).get("value", "")
        text = html_to_text(storage)
        chunks = text_to_chunks(text)
        for i, ch in enumerate(chunks):
            doc_id = f"{page_id}:{i}"
            all_chunks.append(ch)
            metadata.append({
                "doc_id": doc_id,
                "title": title,
                "page_url": urljoin(BASE_URL, f"/pages/viewpage.action?pageId={page_id}")
            })
    logging.info(f"Total chunks to embed: {len(all_chunks)}")

    if not all_chunks:
        logging.warning("No content to index. Exiting.")
        return

    # Batch embedding
    BATCH = 16
    embeddings = []
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i+BATCH]
        logging.debug(f"Embedding batch {i}-{i+len(batch)}")
        embeddings.extend(emb.embed(batch))
        time.sleep(0.05)

    dim = len(embeddings[0])
    recreate_vector_index(r, dim, REDIS_NAMESPACE)

    for md, emb_vec, text in zip(metadata, embeddings, all_chunks):
        store_chunk(r, REDIS_NAMESPACE, md["doc_id"], md["title"], md["page_url"], emb_vec, text)

    logging.info(f"✅ Stored {len(all_chunks)} chunks into Redis namespace {REDIS_NAMESPACE}")


if __name__ == "__main__":
    main()
