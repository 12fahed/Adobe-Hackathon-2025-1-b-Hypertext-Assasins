#!/bin/bash

# Build and run script for PDF processing with Ollama

set -e

echo "Building Docker image with Ollama support..."
docker build --platform linux/amd64 -t pdf-processor-ollama:latest .

echo "Running PDF processor with Ollama..."
echo "Note: This may take several minutes on first run to download the phi model"

# Run with network access for Ollama model download
docker run --rm \
    -v "$(pwd)/app/input:/app/input" \
    -v "$(pwd)/app/output:/app/output" \
    -p 11434:11434 \
    pdf-processor-ollama:latest

echo "Processing complete! Check the ./app/output/ directory for results."
