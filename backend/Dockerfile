FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Run Gunicorn binding dynamically to the $PORT env var (provided by Render)
# Explicitly limit workers to 1 and threads to 4 to optimize memory usage (prevent OOM)
CMD gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:app
