# Use AMD64 compatible Python base image
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

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

# Copy the ML model file
COPY extract-structure-data-model.joblib .

# Copy the app directory structure (but input/output will be mounted)
COPY app/ ./app/

# Set environment variables for better Python behavior in containers
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the main processing script
CMD ["python", "process.py"]
