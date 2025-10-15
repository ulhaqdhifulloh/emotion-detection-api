# main.py
import io
import os
import time
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import urllib.request
import uvicorn 

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Config
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))

# Direktori cache model yang writable & persistent di App Service Linux
# Rekomendasi Microsoft Learn: menulis ke /home
MODEL_DIR = os.getenv("MODEL_DIR", "/home/site/cache/models")
os.makedirs(MODEL_DIR, exist_ok=True)

LOCAL_CKPT = os.path.join(DEPLOY_DIR, "cnn_emotion_model_v6-2.pth")   # jika upload manual (opsional)
DOWNLOAD_CKPT = os.path.join(MODEL_DIR, "model.pth")                  # simpan di /home/site/cache
MODEL_URL = os.getenv("MODEL_URL")                                    # URL unduh langsung (mis. Blob/HF)
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

# Normalisasi pakai nilai standar ImageNet (bisa override via ENV jika perlu)
MEAN = [float(x) for x in os.getenv("MEAN", "0.485,0.456,0.406").split(",")]
STD  = [float(x) for x in os.getenv("STD",  "0.229,0.224,0.225").split(",")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.backends.cudnn.benchmark = True

# Inference lock untuk request paralel
from threading import Lock
_infer_lock = Lock()

# Model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Gunakan weights=None untuk menghindari download saat start
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=device)
    # Dukungan beberapa kemungkinan kunci
    state_keys = ["ema_state_dict", "model_state_dict", "state_dict", "model"]
    state_dict = None
    for k in state_keys:
        if k in ckpt:
            obj = ckpt[k]
            state_dict = obj.state_dict() if hasattr(obj, "state_dict") else obj
            break

    # Pure state_dict
    if state_dict is None and isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt

    if state_dict is None:
        raise RuntimeError("Tidak menemukan state_dict dalam checkpoint. Pastikan kuncinya 'model_state_dict' atau serupa.")

    state_dict = _strip_module_prefix(state_dict)

    class_names = ckpt.get("class_names", ["anger", "fear", "joy", "sad"])
    model = EmotionCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, class_names


# Catatan: verifikasi checksum dihapus untuk kesederhanaan.

def ensure_checkpoint() -> str:
    """Pastikan checkpoint tersedia: gunakan cache di /home dan unduh sekali.

    Perilaku:
    - Jika file sudah ada di DOWNLOAD_CKPT, pakai langsung (startup cepat).
    - Jika belum ada dan MODEL_URL diset, unduh ke DOWNLOAD_CKPT lalu tulis meta.
    - Jika gagal, raise error.
    """
    out_dir = MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)
    meta_file = os.path.join(out_dir, "model.meta")

    # Pakai file cached jika sudah ada
    if os.path.exists(DOWNLOAD_CKPT):
        print(f"[startup] Using cached model: {DOWNLOAD_CKPT}")
        return DOWNLOAD_CKPT

    # Unduh dari MODEL_URL jika tersedia
    if MODEL_URL:
        try:
            print(f"[startup] Downloading model from {MODEL_URL} ...")
            urllib.request.urlretrieve(MODEL_URL, DOWNLOAD_CKPT)
            with open(meta_file, "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"[startup] Model downloaded to {DOWNLOAD_CKPT}")
            return DOWNLOAD_CKPT
        except Exception as e:
            raise RuntimeError(f"Gagal download model: {e}")

    raise RuntimeError("MODEL_URL tidak diset dan model lokal tidak ditemukan.")

# Transform & Face crop
val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# (hapus center-crop opsional yang tidak dipakai)

# Face detector (Haar Cascade)
_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_face_pil(pil_img: Image.Image) -> Image.Image:
    """Crop wajah (margin bawah diperbesar)."""
    img = np.array(pil_img)[:, :, ::-1]  # ke BGR utk OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _haar_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return pil_img

    (x, y, w, h) = faces[0]
    margin_x = int(0.15 * w)   # lebih kecil horizontal
    margin_top = int(0.10 * h)
    margin_bot = int(0.35 * h) # bawah diperbesar (mulut/dagu)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_top)
    x2 = min(img.shape[1], x + w + margin_x)
    y2 = min(img.shape[0], y + h + margin_bot)
    face_crop = img[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

def preprocess_image_pil(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess: face crop, CLAHE, grayscale→RGB, resize+normalize."""
    img = detect_and_crop_face_pil(pil_img)

    # CLAHE pada grayscale lalu kembalikan ke RGB
    g = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    g = clahe.apply(g)
    img = Image.fromarray(g).convert("RGB")

    x = val_tf(img).unsqueeze(0)  # [1,3,H,W]
    return x

# Inference util
@torch.inference_mode()
def predict_logits(model: nn.Module, x: torch.Tensor, use_fp16: bool = False) -> torch.Tensor:
    """
    x: [N,3,H,W] on CPU or GPU.
    """
    x = x.to(device, non_blocking=True)
    if use_fp16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x)
    else:
        return model(x)

def to_response(probs: torch.Tensor, class_names: List[str]) -> Dict[str, Any]:
    conf, idx = probs.max(dim=0)
    entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
    return {
        "emotion": class_names[idx.item()],
        "confidence": float(conf.item()),
        "probs": {class_names[i]: float(probs[i].item()) for i in range(len(class_names))},
        "entropy": entropy,
    }

@torch.inference_mode()
def predict_single(model: nn.Module, pil_img: Image.Image, class_names: List[str], use_tta: bool = False, use_fp16: bool = False) -> Dict[str, Any]:
    """Pipeline preprocess + opsional TTA."""
    if not use_tta:
        x = preprocess_image_pil(pil_img)
        logits = predict_logits(model, x, use_fp16=use_fp16)
        probs = F.softmax(logits, dim=1)[0]
        return to_response(probs, class_names)
    else:
        # TTA ringan: asli + flip horizontal
        views = []
        # original
        views.append(preprocess_image_pil(pil_img))
        # flipped
        views.append(preprocess_image_pil(pil_img.transpose(Image.FLIP_LEFT_RIGHT)))

        x = torch.cat(views, dim=0)  # [2,3,H,W]
        logits = predict_logits(model, x, use_fp16=use_fp16)
        probs = F.softmax(logits, dim=1).mean(dim=0)  # rata-rata
        return to_response(probs, class_names)

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Muat model sekali saat startup
    global MODEL, CLASS_NAMES, CURRENT_CKPT
    print("[startup] Loading checkpoint ...")
    ckpt_path = ensure_checkpoint()
    print(f"[startup] Using checkpoint: {ckpt_path}")
    CURRENT_CKPT = ckpt_path
    MODEL, CLASS_NAMES = load_checkpoint(ckpt_path)
    print("[startup] Model loaded successfully.")
    yield
    # Bersih-bersih sederhana
    MODEL = None

app = FastAPI(title="Emotion Detection API", version="1.0.0", lifespan=lifespan)

# CORS
_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
_origins_list = [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Emotion detection API is running!"}

class PredictResult(BaseModel):
    emotion: str
    confidence: float
    probs: Dict[str, float]
    entropy: float
    latency_ms: float
    device: str
    tta_used: bool
    fp16_used: bool

class BatchItem(BaseModel):
    filename: str
    result: Optional[PredictResult] = None
    error: Optional[str] = None

class BatchResponse(BaseModel):
    items: List[BatchItem]

# Muat model saat startup
MODEL: Optional[nn.Module] = None
CLASS_NAMES: List[str] = []
CURRENT_CKPT: str = ""

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "classes": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "checkpoint": CURRENT_CKPT
    }

@app.post("/predict", response_model=PredictResult)
def predict_endpoint(
    file: UploadFile = File(...),
    tta: bool = Query(False, description="Gunakan TTA flip horizontal"),
    fp16: bool = Query(True, description="Autocast FP16 saat GPU")
):
    global MODEL, CLASS_NAMES
    start = time.perf_counter()
    try:
        contents = file.file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Gagal membaca gambar: {e}")
    finally:
        file.file.close()

    with _infer_lock:
        out = predict_single(MODEL, pil, CLASS_NAMES, use_tta=tta, use_fp16=fp16)

    latency = (time.perf_counter() - start) * 1000.0
    return {
        **out,
        "latency_ms": round(latency, 2),
        "device": str(device),
        "tta_used": bool(tta),
        "fp16_used": bool(fp16 and (device.type == "cuda"))
    }

@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch_endpoint(
    files: List[UploadFile] = File(..., description="Kirim beberapa file sekaligus"),
    tta: bool = Query(False),
    fp16: bool = Query(True)
):
    global MODEL, CLASS_NAMES
    items: List[BatchItem] = []

    # Untuk efisiensi, kita tetap proses per-file karena ada face-crop/CLAHE per gambar
    for f in files:
        try:
            contents = f.file.read()
            pil = Image.open(io.BytesIO(contents)).convert("RGB")
            start = time.perf_counter()
            with _infer_lock:
                out = predict_single(MODEL, pil, CLASS_NAMES, use_tta=tta, use_fp16=fp16)
            latency = (time.perf_counter() - start) * 1000.0
            items.append(BatchItem(
                filename=f.filename,
                result=PredictResult(
                    **out,
                    latency_ms=round(latency, 2),
                    device=str(device),
                    tta_used=bool(tta),
                    fp16_used=bool(fp16 and (device.type == "cuda"))
                )
            ))
        except Exception as e:
            items.append(BatchItem(filename=f.filename, error=str(e)))
        finally:
            try:
                f.file.close()
            except Exception:
                pass

    return BatchResponse(items=items)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)