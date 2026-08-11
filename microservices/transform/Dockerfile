FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Orgpedia docint uses PyTorch, which we install via cu121 index to match our deployment target
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
# Need vendor dir for orgpedia local installation
COPY vendor vendor/
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p input output

# Copy the rest of the application (app.py, word_recognizer.py, src/, conf/)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
