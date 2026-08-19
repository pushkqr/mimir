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
# pip 23.0.1 (what python:3.10-slim ships) compares a wheel's metadata Name byte-for-byte
# against the requested name, so it discards `typing_extensions`/`Jinja2` wheels as
# "inconsistent Name" and falls back to their sdists - whose build deps (flit_core) don't exist
# on the PyTorch index. Upgrading pip restores name normalisation; the extra index is a
# backstop so any remaining source build can still reach PyPI for its build deps. The explicit
# +cu121 pin is what keeps that extra index safe: that local version only exists on PyTorch's
# index, so PyPI can't shadow it with a newer CUDA build.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
