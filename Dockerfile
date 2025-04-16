FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for PDF processing and model training
RUN pip install --no-cache-dir PyPDF2==3.0.1 transformers==4.30.2 torch==2.0.1 datasets==2.13.1 tqdm

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV USE_GPU=False
ENV MODEL_PATH=/app/models/final_model
ENV MAX_SEQ_LENGTH=512
ENV BATCH_SIZE=32
ENV NUM_WORKERS=4

# Create necessary directories
RUN mkdir -p /app/models /app/data/processed_data

# Command to run when container starts
CMD ["python", "core/transformer.py"]

# Expose port for API
EXPOSE 8000