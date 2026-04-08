FROM public.ecr.aws/docker/library/python:3.11-slim

# Force stdout and stderr to be unbuffered
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies (git is often needed for openenv)
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Set Port
ENV PORT=7860
EXPOSE 7860

# Run with -u to double-ensure unbuffered logging
CMD ["python", "-u", "inference.py"]