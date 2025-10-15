# Emotion Detection API (FastAPI) — Deploy Azure App Service (tanpa Docker)

Server FastAPI untuk menyajikan model emosi PyTorch. Folder ini siap di-deploy ke Azure App Service Linux (Publish: Code) menggunakan `gunicorn` + worker `uvicorn`.

## Fitur

- Endpoint: `GET /health`, `POST /predict` (gambar tunggal), `POST /predict-batch` (banyak gambar)
- Preprocessing: crop wajah (Haar Cascade), CLAHE, grayscale→RGB, resize+normalize
- Opsi TTA (flip horizontal) dan FP16 autocast di GPU
- CORS dapat dikonfigurasi via environment variable (`CORS_ALLOW_ORIGINS`)

## Checklist Cepat — Azure App Service (tanpa Docker)

1) Buat Web App
- Portal: Create Resource → Web App
- Publish: `Code`
- Runtime stack: `Python 3.11`
- Region: dekat kamu (mis. Southeast Asia)
- Plan: Free/Basic (pakai credit Student)

2) Hubungkan Source
- Deployment: pilih GitHub → repo kamu → branch `main`
- Alternatif: ZIP deploy manual (Upload → Deploy)

3) App Settings (Environment Variables)
- Portal → Web App → Configuration → Application settings → Add:
  - `MODEL_URL` = `https://emotiondetectionstorage.blob.core.windows.net/models/cnn_emotion_model_v6-2.pth`
  - (opsional) `IMG_SIZE` = `224`
  - (opsional) `CORS_ALLOW_ORIGINS` = `*`
- Save dan restart jika diminta.

4) Startup Command
- Portal → Configuration → General settings → Startup Command:
  - Jika `main.py` di root: `gunicorn -k uvicorn.workers.UvicornWorker -w 2 main:app`
  - Jika `main.py` di subfolder (mis. `render-deploy/`): `gunicorn -k uvicorn.workers.UvicornWorker -w 2 render-deploy.main:app`
- `gunicorn` + worker `uvicorn` adalah cara resmi/cepat di App Service Linux.

## Requirements (CPU, App Service Linux)

File `requirements.txt` telah disesuaikan:

```
fastapi==0.104.1
uvicorn==0.24.0
gunicorn
python-multipart==0.0.6
pydantic==2.5.0
torch==2.3.1
torchvision==0.18.1
pillow
opencv-python-headless==4.8.1.78
numpy==1.24.3
```

- `gunicorn` dibutuhkan karena Startup Command memakai `gunicorn`.
- `opencv-python-headless` menghindari dependency GUI.
- Versi `torch/torchvision` disesuaikan dengan versi training (contoh 2.3.1/0.18.1).
- Python 3.11: pilih saat membuat Web App (App Service mengatur runtime via portal; `runtime.txt` tidak dipakai App Service).

## Environment Variables yang didukung

- `MODEL_URL`: tautan unduh langsung model (Blob/Hugging Face). Jika Hugging Face, gunakan format URL `.../resolve/...`.
- `IMG_SIZE`: default `224`
- `CORS_ALLOW_ORIGINS`: default `*` (boleh dipisah koma untuk beberapa origin)

Catatan path Azure: aplikasi menyimpan model terunduh ke folder writable yang persisten di App Service Linux: `/home/site/cache/models/model.pth`.

## Jalankan Lokal (opsional)

1. Buat dan aktifkan virtual environment.
2. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
3. Set env dan mulai server dev:
   - Windows PowerShell:
     ```powershell
     $env:MODEL_URL = "https://huggingface.co/username/emotion-model/resolve/main/model.pth"
     uvicorn main:app --host 0.0.0.0 --port 3003
     ```
   - macOS/Linux:
     ```bash
     export MODEL_URL="https://huggingface.co/username/emotion-model/resolve/main/model.pth"
     uvicorn main:app --host 0.0.0.0 --port 3003
     ```

## Uji Cepat (curl)

```bash
curl http://localhost:3003/health

curl -X POST "http://localhost:3003/predict?tta=false&fp16=true" \
  -F "file=@./../images/test/joy_WIN_20251009_00_23_41_Pro.jpg"
```

## Endpoint API

- `GET /health`: info server, device, kelas, dan konfigurasi.
- `POST /predict`: form-data `file`, query `tta` (bool), `fp16` (bool)
- `POST /predict-batch`: form-data `files[]`, query `tta`, `fp16`

## Troubleshooting

- "Checkpoint not found": pastikan `MODEL_URL` valid dan dapat diakses (bisa Blob atau Hugging Face `.../resolve/...`).
- Unggahan gambar besar: gunakan JPEG dan batasi ukuran di sisi klien.
- Error CORS: set `CORS_ALLOW_ORIGINS` ke origin situs Anda (mis. `http://localhost:5500`) atau `*` jika publik.

---

Folder ini difokuskan untuk deploy ke Azure App Service.