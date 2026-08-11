FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch installed separately, pinned to an explicit CUDA 12.1 build via PyTorch's own
# version-specific index. An unpinned `torch` in requirements.txt grabbed whatever was newest
# on PyPI at build time (found: 2.13.0+cu130, requiring CUDA 13.0), which this host's driver
# (575.57.08, max CUDA 12.9) can't satisfy - torch.cuda.is_available() came back False despite
# the GPU being attached to the container. cu121 is comfortably within range for any driver
# reporting 12.x+ support.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
