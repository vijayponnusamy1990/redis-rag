FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose port 8080
EXPOSE 8080

# Start FastAPI with Uvicorn on port 8080
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8080"]