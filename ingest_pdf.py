"""
Ingest PDF documents, chunk, embed with Ollama or Azure OpenAI, and store in Redis (RediSearch vector index).

- EMBEDDING_BACKEND=ollama → embeddings via local Ollama
- EMBEDDING_BACKEND=azure → embeddings via Azure OpenAI
- Redis vector index is dropped and recreated on each run
"""

import os
import time
import logging
import numpy as np
import redis
import requests
from dotenv import load_dotenv
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

import tiktoken
from pypdf import PdfReader
from openai import AzureOpenAI

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Config
PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_NAMESPACE = os.getenv("REDIS_NAMESPACE", "pdf:rdb")
REDIS_INDEX = os.getenv("REDIS_VECTOR_INDEX", "pdf_idx")

# Embedding backend
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "ollama").lower()

# Ollama config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Azure OpenAI config
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))


# --- Utils ---
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


# --- Embedding Backends ---
class EmbeddingBackend:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OllamaEmbeddingBackend(EmbeddingBackend):
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


class AzureOpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self):
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise ValueError("Azure OpenAI configuration missing in environment")
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        logging.debug(f"Requesting Azure OpenAI embeddings for {len(texts)} texts")
        response = self.client.embeddings.create(
            model=AZURE_OPENAI_EMBED_MODEL,
            input=texts
        )
        return [d.embedding for d in response.data]


class EmbeddingBackendFactory:
    @staticmethod
    def create() -> EmbeddingBackend:
        if EMBEDDING_BACKEND == "ollama":
            logging.info("Using Ollama embedding backend")
            return OllamaEmbeddingBackend()
        elif EMBEDDING_BACKEND == "azure":
            logging.info("Using Azure OpenAI embedding backend")
            return AzureOpenAIEmbeddingBackend()
        else:
            raise ValueError(f"Unsupported EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")


# --- Redis ---
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
        TextField("file_path"),
        TextField("page_num"),
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


def store_chunk(r: redis.Redis, prefix: str, doc_id: str, title: str, file_path: str, page_num: int, embedding: List[float], text: str):
    vec = np.array(embedding, dtype=np.float32).tobytes()
    key = f"{prefix}:{doc_id}"
    r.hset(key, mapping={
        "title": title,
        "file_path": file_path,
        "page_num": str(page_num),
        "text": text,
        "embedding": vec,
    })
    logging.debug(f"Stored chunk {doc_id} with {len(embedding)}-dim embedding")


def extract_text_from_pdf(file_path: str) -> List[tuple]:
    """Return list of (page_num, text)"""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
        except Exception as e:
            logging.warning(f"Failed to read page {i} of {file_path}: {e}")
    return pages


# --- Main ---
def main():
    logging.info("Starting PDF ingestion...")
    emb = EmbeddingBackendFactory.create()
    r = get_redis_client()

    all_chunks, metadata = [], []

    for fname in os.listdir(PDF_FOLDER):
        logging.debug(f"Checking file: {fname}")
        if not fname.lower().endswith(".pdf"):
            logging.debug(f"Skipping non-PDF file: {fname}")
            continue
        fpath = os.path.join(PDF_FOLDER, fname)
        logging.info(f"Processing PDF: {fpath}")
        pages = extract_text_from_pdf(fpath)
        logging.debug(f"Extracted {len(pages)} pages from {fpath}")

        for page_num, text in pages:
            logging.debug(f"Processing page {page_num} of {fpath}")
            chunks = text_to_chunks(text)
            logging.debug(f"Split page {page_num} into {len(chunks)} chunks")
            for i, ch in enumerate(chunks):
                doc_id = f"{os.path.basename(fpath)}:{page_num}:{i}"
                all_chunks.append(ch)
                metadata.append({
                    "doc_id": doc_id,
                    "title": os.path.basename(fpath),
                    "file_path": fpath,
                    "page_num": page_num
                })
                logging.debug(f"Prepared chunk {doc_id} for embedding")

    logging.info(f"Total chunks to embed: {len(all_chunks)}")
    if not all_chunks:
        logging.warning("No content to index. Exiting.")
        return

    # Batch embedding
    BATCH = 16
    embeddings = []
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i+BATCH]
        logging.debug(f"Embedding batch {i}-{i+len(batch)} (size {len(batch)})")
        batch_embeddings = emb.embed(batch)
        logging.debug(f"Received {len(batch_embeddings)} embeddings for batch {i}-{i+len(batch)}")
        embeddings.extend(batch_embeddings)
        time.sleep(0.05)

    dim = len(embeddings[0])
    logging.debug(f"Embedding dimension: {dim}")
    recreate_vector_index(r, dim, REDIS_NAMESPACE)

    for md, emb_vec, text in zip(metadata, embeddings, all_chunks):
        logging.debug(f"Storing chunk {md['doc_id']} in Redis")
        store_chunk(r, REDIS_NAMESPACE, md["doc_id"], md["title"], md["file_path"], md["page_num"], emb_vec, text)

    logging.info(f"✅ Stored {len(all_chunks)} chunks into Redis namespace {REDIS_NAMESPACE}")


if __name__ == "__main__":
    main()
