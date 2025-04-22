FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /serve

# Copy pyproject.toml
COPY pyproject.toml .

# Create virtual environment and install deps with uv
RUN uv sync

# Copy the FastAPI app
COPY serve/ ./serve
# Copy the tensorflow models
COPY models/ ./models
COPY data/vocab.json ./data/vocab.json
# Expose the port
EXPOSE 8000

# Run the app with uvicorn from .venv
CMD [".venv/bin/uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
