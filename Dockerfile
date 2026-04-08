FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Set port (HF uses 7860)
ENV PORT=7860

# ⚠️ MUST be static
EXPOSE 7860

# Start server first, then inference
CMD sh -c "python -m server.app & sleep 5 && python inference.py"