# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Note: Removed redis-server because we are using the managed Railway Redis service
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
# Note: Kept the timeout increase to prevent build failures
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code
COPY . .

# Create modules directory if it doesn't exist
RUN mkdir -p modules

# Run the application
# CRITICAL FIX: We use the shell command format (no brackets) so $PORT is expanded correctly
CMD uvicorn main:app --host 0.0.0.0 --port $PORT