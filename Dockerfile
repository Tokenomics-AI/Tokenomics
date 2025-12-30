# Tokenomics Platform Docker Image
# Linux-based container with LLMLingua support

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt

# Copy the application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command - run tests
CMD ["python", "-c", "from tokenomics.core import TokenomicsPlatform; print('Platform initialized successfully')"]





