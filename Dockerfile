FROM public.ecr.aws/docker/library/python:3.11-slim

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

CMD sh -c "python -m server.app & sleep 5 && python inference.py"
