# Gunakan image dasar Python slim (hemat tapi stabil)
FROM python:3.11.9-slim

# Buat direktori kerja
WORKDIR /app

# Copy semua file (kode + model)
COPY . .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Pastikan OpenCV bisa jalan (headless)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Expose port default (App Service akan ganti ke $PORT otomatis)
EXPOSE 8000

# Jalankan API pakai gunicorn + uvicorn worker
CMD gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:${PORT:-8000} main:app