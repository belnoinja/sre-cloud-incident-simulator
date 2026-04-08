FROM public.ecr.aws/docker/library/python:3.11-slim

# Force logs to flush immediately
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies for openenv-core
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# HF Standard Port
ENV PORT=7860
EXPOSE 7860

# MANDATORY: Start the server, NOT the inference script
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]