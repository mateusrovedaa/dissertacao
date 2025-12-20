# Multi-stage build for optimized Docker images
# Base stage with common dependencies
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Create logs directory
RUN mkdir -p logs

# =============================================================================
# Edge stage - only edge-related files
FROM base AS edge

# Copy compressors module (required by edge)
COPY compressors/ ./compressors/

# Copy edge application
COPY vispac_edge_prototype.py .
COPY compressors.py .

# Copy datasets
COPY datasets/ ./datasets/

# Copy configuration
COPY config/ ./config/

CMD ["python", "vispac_edge_prototype.py"]

# =============================================================================
# Fog stage - only fog-related files
FROM base AS fog

# Copy compressors module (required by fog for decompression)
COPY compressors/ ./compressors/

# Copy fog application
COPY news2_api.py .
COPY compressors.py .

CMD ["python", "news2_api.py"]

# =============================================================================
# Cloud stage - only cloud-related files
FROM base AS cloud

# Copy cloud application
COPY cloud_api.py .

CMD ["python", "cloud_api.py"]
