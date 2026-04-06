FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install uv (for lockfile support)
RUN pip install uv

# Install dependencies using pyproject.toml
RUN pip install .

# Expose port (required for HF Spaces)
EXPOSE 7860

# Run FastAPI app from required server entrypoint
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]