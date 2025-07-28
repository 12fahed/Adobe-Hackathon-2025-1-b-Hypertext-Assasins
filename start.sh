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
