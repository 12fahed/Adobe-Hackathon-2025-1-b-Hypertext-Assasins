# Use AMD64 compatible Python base image
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Ollama
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary Python scripts
COPY process.py .
COPY extract_data_from_pdf.py .
COPY ml-model.py .
COPY generate_query_output.py .

# Copy input configuration files
COPY challenge1b_input.json .

# Copy the ML model file
COPY extract-structure-data-model.joblib .

# Copy the app directory structure (but input/output will be mounted)
COPY app/ ./app/

# Set environment variables for better Python behavior in containers
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose Ollama port (optional, for debugging)
EXPOSE 11434

# Use the startup script as the entry point
CMD ["/app/start.sh"]
