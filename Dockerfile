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

# Create a startup script to handle Ollama initialization
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "=== PDF Processing with Ollama Integration ==="

# Function to check if Ollama is responding
check_ollama() {
    curl -s http://localhost:11434/api/tags >/dev/null 2>&1
}

# Start Ollama service
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in {1..30}; do
    if check_ollama; then
        echo "Ollama is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Timeout waiting for Ollama to start"
        exit 1
    fi
    sleep 2
done

# Check if phi model exists, if not download it
echo "Checking for phi model..."
if ! ollama list | grep -q "phi:latest"; then
    echo "Downloading phi model (this may take several minutes)..."
    ollama pull phi
    echo "Phi model downloaded successfully!"
else
    echo "Phi model already available!"
fi

# Run the main PDF processing pipeline
echo "Starting PDF processing pipeline..."
python process.py

# Check if AI analysis should run
if [ -f "challenge1b_input.json" ] && [ -s "challenge1b_input.json" ]; then
    echo "Found challenge1b_input.json, running AI analysis..."
    python generate_query_output.py
    echo "AI analysis complete!"
else
    echo "No challenge1b_input.json found. Skipping AI analysis."
fi

echo "=== Processing Complete ==="

# Keep container running if needed for debugging
# tail -f /dev/null
EOF

RUN chmod +x /app/start.sh

# Expose Ollama port (optional, for debugging)
EXPOSE 11434

# Use the startup script as the entry point
CMD ["/app/start.sh"]
