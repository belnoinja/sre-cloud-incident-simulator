FROM python:3.11-slim

WORKDIR /app

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy logic
COPY . .

# Expose port (HF spaces default is 7860, but standard uvicorn is 8000. OpenEnv handles via FastAPI)
EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
