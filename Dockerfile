FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG TORCH_CUDA_FLAVOR=cu126
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh

RUN python -m pip install --upgrade pip "setuptools<82" wheel

# Surya runtime venv
RUN python -m venv /opt/venvs/surya && \
    /opt/venvs/surya/bin/python -m pip install --upgrade pip "setuptools<82" wheel && \
    /opt/venvs/surya/bin/python -m pip install /app && \
    /opt/venvs/surya/bin/python -m pip install \
      "pypdf>=4.0" \
      "reportlab>=4.0" \
      "surya-ocr==0.17.1" \
      "transformers==4.57.1" \
      "huggingface-hub>=0.34,<1.0" \
      "tokenizers>=0.22,<0.23" \
      "pypdfium2==4.30.0" && \
    /opt/venvs/surya/bin/python -m pip install \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_FLAVOR}" \
      --upgrade --force-reinstall \
      "torch==${TORCH_VERSION}+${TORCH_CUDA_FLAVOR}" \
      "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_FLAVOR}" \
      "torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA_FLAVOR}" && \
    /opt/venvs/surya/bin/python -m pip install --upgrade "pillow>=10.2,<11.0"

# Chandra runtime venv
RUN python -m venv /opt/venvs/chandra && \
    /opt/venvs/chandra/bin/python -m pip install --upgrade pip "setuptools<82" wheel && \
    /opt/venvs/chandra/bin/python -m pip install /app && \
    /opt/venvs/chandra/bin/python -m pip install \
      "pypdf>=4.0" \
      "reportlab>=4.0" \
      "chandra-ocr[hf]==0.2.0" \
      "pypdfium2==4.30.0" && \
    /opt/venvs/chandra/bin/python -m pip install \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_FLAVOR}" \
      --upgrade --force-reinstall \
      "torch==${TORCH_VERSION}+${TORCH_CUDA_FLAVOR}" \
      "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_FLAVOR}" \
      "torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA_FLAVOR}"

RUN sed -i 's/\r$//' /app/scripts/docker-entrypoint.sh && \
    chmod +x /app/scripts/docker-entrypoint.sh && \
    mkdir -p /cache/hf_chandra /cache/hf_surya /cache/surya_models /cache/modelscope /data/work /data/in /data/out

ENV PATH="/opt/venvs/chandra/bin:/opt/venvs/surya/bin:${PATH}" \
    UNISCAN_CHANDRA_PYTHON="/opt/venvs/chandra/bin/python" \
    UNISCAN_SURYA_PYTHON="/opt/venvs/surya/bin/python" \
    UNISCAN_CHANDRA_HF_HOME="/cache/hf_chandra" \
    UNISCAN_CHANDRA_HUGGINGFACE_HUB_CACHE="/cache/hf_chandra/hub" \
    UNISCAN_CHANDRA_HF_HUB_CACHE="/cache/hf_chandra/hub" \
    UNISCAN_SURYA_HF_HOME="/cache/hf_surya" \
    UNISCAN_SURYA_HUGGINGFACE_HUB_CACHE="/cache/hf_surya/hub" \
    UNISCAN_SURYA_HF_HUB_CACHE="/cache/hf_surya/hub" \
    UNISCAN_SURYA_MODEL_CACHE_DIR="/cache/surya_models" \
    UNISCAN_SURYA_MODELSCOPE_CACHE="/cache/modelscope" \
    UNISCAN_CHANDRA_DEVICE_POLICY="auto" \
    UNISCAN_CHANDRA_PREFER_GPU="1" \
    UNISCAN_CHANDRA_REQUIRE_GPU="0" \
    UNISCAN_SURYA_ALLOW_TEXT_FALLBACK="0" \
    UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON="1" \
    HF_HUB_DISABLE_SYMLINKS_WARNING="1" \
    UNISCAN_WORK_ROOT="/data/work" \
    UNISCAN_DEFAULT_LANG="rus+eng" \
    UNISCAN_HTTP_PORT="8000"

VOLUME ["/cache", "/data/work", "/data/in", "/data/out"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${UNISCAN_HTTP_PORT:-8000}/health" || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD []
