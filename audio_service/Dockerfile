# ─────────────────────────────────────────────────────────────────────────────
# Faster-Whisper Microservice – CPU-optimised Docker image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies (libgomp is needed by ctranslate2 on Linux)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model (Commented out to prevent build hangs/timeout)
# RUN python - <<'EOF'
# from huggingface_hub import snapshot_download
# import os
# model = os.environ["WHISPER_MODEL"]
# repo  = f"Systran/faster-whisper-{model}"
# print(f"Pre-downloading model: {repo}")
# snapshot_download(repo_id=repo)
# print("Done.")
# EOF

# Copy application source
COPY main.py .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
