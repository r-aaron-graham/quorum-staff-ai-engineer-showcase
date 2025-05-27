### Dockerfile for Quorum AI Services ###

# Base image with Python 3.10
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt ./

# Install system dependencies and Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ ./src
COPY infra/docker/docker-entrypoint.sh ./docker-entrypoint.sh

# Make entrypoint executable
RUN chmod +x ./docker-entrypoint.sh

# Expose port
EXPOSE 8000

# Entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]
