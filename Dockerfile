FROM public.ecr.aws/docker/library/python:3.11-slim

# Prevent Python from buffering logs (Critical for seeing tracebacks)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files (inference.py, client.py, models.py)
COPY . .

# Set standard port
ENV PORT=7860
EXPOSE 7860

# Run in unbuffered mode
CMD ["python", "-u", "inference.py"]