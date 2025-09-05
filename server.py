# server.py
from fastapi import FastAPI
from pymongo import MongoClient
from bson.decimal128 import Decimal128   # ✅ add this
import torch
import torch.nn as nn
import torchvision.models as models
import joblib
import numpy as np
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


# =============================
# Env & Mongo
# =============================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "pv_forecast")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# =============================
# Model definition (matches checkpoint shapes)
# CNN fc -> 128; LSTM input 129 (128 + 1 PV); hidden 64; 2 layers; FC 64->1
# =============================
class SkyForecastNet(nn.Module):
    def __init__(self, cnn_out=128, hidden_size=64, num_layers=2):
        super().__init__()
        # ResNet18 backbone; fc changed to 512->128 so 'cnn.fc.*' matches checkpoint
        self.cnn = models.resnet18(weights=None)
        in_features = self.cnn.fc.in_features  # 512 for resnet18
        self.cnn.fc = nn.Linear(in_features, cnn_out)  # <- 512 -> 128

        # LSTM expects 128 (image feat) + 1 (raw PV) = 129
        self.lstm = nn.LSTM(
            input_size=cnn_out + 1,
            hidden_size=hidden_size,  # 64
            num_layers=num_layers,    # 2
            batch_first=True
        )

        # Final prediction head
        self.fc = nn.Linear(hidden_size, 1)  # 64 -> 1

    def forward(self, x_seq, pv_seq):
        """
        x_seq:  (B, T, C, H, W)
        pv_seq: (B, T, 1)  -- scaled PV
        """
        B, T, C, H, W = x_seq.size()
        feats = []
        for t in range(T):
            img_feat = self.cnn(x_seq[:, t, :, :, :])     # (B, 128)
            pv_t = pv_seq[:, t, :]                         # (B, 1)
            step = torch.cat([img_feat, pv_t], dim=1)      # (B, 129)
            feats.append(step.unsqueeze(1))                # (B, 1, 129)

        feats = torch.cat(feats, dim=1)                    # (B, T, 129)
        lstm_out, _ = self.lstm(feats)                     # (B, T, 64)
        y = self.fc(lstm_out[:, -1, :])                    # (B, 1) last step
        return y

# =============================
# FastAPI
# =============================
app = FastAPI(title="Sky Forecasting Inference API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict: ["http://127.0.0.1:5501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Globals for model + scaler
# =============================
model = None
scaler = None

@app.on_event("startup")
async def load_artifacts():
    """Load model & scaler once on startup"""
    global model, scaler
    print("🔄 Loading model & scaler...")
    model = SkyForecastNet()
    state_dict = torch.load("saved_model.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    scaler = joblib.load("scaler.pkl")
    print("✅ Model & scaler ready")

# =============================
# Image preprocessing
# =============================
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))  # typical for ResNet
def preprocess_image(img_bytes: bytes) -> np.ndarray:
    # Bytes -> PIL -> resize -> float tensor CHW
    image = Image.open(BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0     # HWC, [0,1]
    chw = np.transpose(arr, (2, 0, 1))                    # CHW
    return chw

# =============================
# Predict Endpoint
# =============================
@app.get("/predict")
def predict():
    if model is None or scaler is None:
        return {"error": "Model or scaler not loaded"}

    # Pull latest up to 8 docs, oldest->newest order to form a temporal sequence
    docs = list(collection.find().sort("createdAt", -1).limit(8))[::-1]
    if not docs:
        return {"error": "No documents in queue"}

    imgs = []
    pvs = []
    for d in docs:
        img_bytes = bytes(d["image"])
        chw = preprocess_image(img_bytes)
        imgs.append(chw)

        pv_raw = d["pv"]
        if isinstance(pv_raw, Decimal128):
            pv_val = float(pv_raw.to_decimal())
        else:
            pv_val = float(pv_raw)
        pvs.append(pv_val)

    T = len(imgs)
    x_seq = torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32).unsqueeze(0)

    # Scale PVs
    pv_scaled = scaler.transform(np.array(pvs, dtype=np.float32).reshape(-1, 1))
    pv_seq = torch.tensor(pv_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        y_scaled = model(x_seq, pv_seq).item()

    y_real = scaler.inverse_transform(np.array([[y_scaled]])).item()

    return {
        "queue_size": T,
        "prediction_scaled": float(y_scaled),
        "prediction_real": float(y_real),
        "latest_createdAt": docs[-1]["createdAt"],
    }

# =============================
# Health Check
# =============================
@app.get("/health")
def health():
    try:
        _ = collection.estimated_document_count()
        ok_db = True
    except Exception:
        ok_db = False
    return {"ok": True, "db": ok_db}
