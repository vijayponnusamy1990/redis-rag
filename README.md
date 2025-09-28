# Redis RAG API Setup

This project provides a FastAPI REST service for RAG (Retrieval Augmented Generation) using Redis as a vector database, Ollama for embeddings/LLMs, and Confluence ingestion.

## Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup (Python Local)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd redis-rag
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Start Redis locally (optional)**
   ```bash
   docker run -d --name redis-server -p 6379:6379 redis:7
   ```

5. **Start Ollama locally (optional)**
   ```bash
   docker run -d --name ollama -p 11434:11434 ollama/ollama
   docker exec ollama ollama pull llama2
   docker exec ollama ollama pull nomic-embed-text
   ```

6. **Run the FastAPI app**
   ```bash
   uvicorn rag_api:app --host 0.0.0.0 --port 8080
   ```

## Setup (Docker Compose)

1. **Build and start all services**
   ```bash
   docker compose up --build
   ```

   This will:
   - Start Redis on port 6379
   - Start Ollama on port 11434 and pull `llama2` and `nomic-embed-text` models
   - Start the FastAPI app on port 8080

## Sample Requests

### 1. RAG Query Endpoint

**POST** `/rag`

```bash
curl -X POST "http://localhost:8080/rag" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is periodic table?", "history": []}'
```

**Response:**
```json
{
  "answer": "..."
}
```

### 2. Ingest Confluence Endpoint

**POST** `/ingest/confluence`

```bash
curl -X POST "http://localhost:8080/ingest/confluence"
```

**Response:**
```json
{
  "status": "Ingestion triggered"
}
```

## API Documentation

Once running, visit [http://localhost:8080/docs](http://localhost:8080/docs) for interactive Swagger docs.

---

**Note:**  
- Ensure your `.env` file is configured with correct Redis and Ollama URLs.
- Ollama models are pulled automatically via Docker Compose.
- For production, secure your endpoints and credentials.
