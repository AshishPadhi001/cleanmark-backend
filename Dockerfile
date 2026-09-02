# CleanMark AI Backend Dockerfile (Ultra-Lightweight Mathematical Engine)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (FFmpeg for fast video remuxing & Curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight Python dependencies (NumPy, OpenCV Headless, Pillow - No PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy reference alpha templates and backend application source code
COPY models ./models
COPY app ./app
COPY run.py .

# Create storage directories
RUN mkdir -p /app/storage/uploads /app/storage/outputs /app/storage/videos /app/storage/temp

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI application
CMD ["python", "run.py"]
