FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including ffmpeg for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Provide a default PORT but allow it to be overridden
ENV PORT=10000
EXPOSE $PORT

# Run via sh so the PORT environment variable expands correctly for Render/Fly.io
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT"]
