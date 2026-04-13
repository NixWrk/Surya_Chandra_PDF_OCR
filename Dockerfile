FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -e ".[ocr]" && \
    python -m pip install --no-cache-dir \
      "surya-ocr" \
      "chandra-ocr[hf]" \
      "requests" \
      "transformers==4.57.1" \
      "tokenizers==0.22.1" \
      "huggingface-hub==0.34.4"

EXPOSE 8000

CMD ["python", "-m", "uniscan", "serve-http", "--host", "0.0.0.0", "--port", "8000"]
