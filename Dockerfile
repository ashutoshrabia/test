# Start with a lightweight Python image
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for cache directories
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV HF_HOME=/tmp/hf_home
ENV XDG_CACHE_HOME=/tmp/cache

RUN mkdir -p /tmp/cache /tmp/hf_home
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]