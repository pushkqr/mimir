FROM python:3.10-slim

WORKDIR /app

# word_recognizer.py's own @Vision.factory declaration lists apt:tesseract-ocr-all as a
# dependency (line 24) - it was never actually installed here, so Tesseract had only its
# English data file and produced Latin-lookalike garbage on Devanagari script (word_recognizer
# already requests languages=["eng","mar"], so this isn't a code bug, just a missing package).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git tesseract-ocr tesseract-ocr-all poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# uv, not pip: faster resolve/install for this dependency-heavy service. Not load-bearing for
# correctness - the actual build failure this repo hit (FileNotFoundError on
# orgpedia/ledger/old_ledger.py) turned out to be a broken symlink committed upstream in
# orgpedia/orgpedia, pointing outside the repo entirely (../../../mahgetNewGRs/.bak/ledger.py).
# It reproduced identically under plain pip too; removing the dead symlink from our vendored
# orgpedia sdist (vendor/orgpedia-0.1.0.tar.gz) is what actually fixed it.
RUN pip install --no-cache-dir uv

COPY requirements.txt .
# Orgpedia docint uses PyTorch, which we install via cu121 index to match our deployment target
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cu121
# Need vendor dir for orgpedia local installation
COPY vendor vendor/
RUN uv pip install --system --no-cache -r requirements.txt

# Create necessary directories - logs/ is where docint's pdftable_finder task writes its
# per-document table-extraction log; it assumes the directory already exists and doesn't
# create it itself, so /transform 500s with a bare FileNotFoundError without this.
RUN mkdir -p input output logs

# Copy the rest of the application (app.py, word_recognizer.py, src/, conf/)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
