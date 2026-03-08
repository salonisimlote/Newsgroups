FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create directories for generated artifacts
RUN mkdir -p embeddings vector_store data clustering cache

# Expose port
EXPOSE 8000

# Start with uvicorn on port 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
