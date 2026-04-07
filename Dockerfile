FROM python:3.11.9-slim

WORKDIR /app

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy logic
COPY . .

# Hugging Face provides PORT environment variable (default 7860)
ENV PORT=7860
EXPOSE $PORT

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
