FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Required for some python package builds
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the local modules are copied
COPY . .

# HF/Evaluator standard port
ENV PORT=7860
EXPOSE 7860

# Force unbuffered output so logs show up immediately in the participant log
ENV PYTHONUNBUFFERED=1

CMD ["python", "inference.py"]