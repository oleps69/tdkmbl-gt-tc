# Dockerfile (kopyala - yapıştır)
FROM python:3.10-slim

# Sistem bağımlılıkları — onnxruntime için libgomp1 dahil
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ÖNEMLİ: pip'i güncelle, sonra requirements yükle
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
