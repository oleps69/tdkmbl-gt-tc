import os
import cv2
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File
from insightface.app import FaceAnalysis

app = FastAPI()

# =========================
# GLOBALS
# =========================
face_app = None
embedding_db = {}
MODEL_READY = False
DB_PATH = "face_embeddings.pkl"
IMAGE_SIZE = (640, 640)
THRESHOLD = 0.60

# =========================
# MODEL LOADER
# =========================
def load_model():
    global face_app, MODEL_READY
    if face_app is None:
        print("🔵 Lazy-loading InsightFace model...")
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=0, det_size=IMAGE_SIZE)
        MODEL_READY = True
        print("✅ Model ready")

def load_embedding_db():
    global embedding_db
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            embedding_db = pickle.load(f)
        print(f"📦 Loaded {len(embedding_db)} identities")
    else:
        print("⚠️ face_embeddings.pkl not found")

# =========================
# STARTUP
# =========================
@app.on_event("startup")
def startup():
    load_embedding_db()

# =========================
# UTILS
# =========================
def extract_embedding(img):
    faces = face_app.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    emb = face.embedding.astype(np.float32)
    return emb / np.linalg.norm(emb)

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def predict_identity(emb):
    best_name = "unknown"
    best_score = -1

    for name, ref_emb in embedding_db.items():
        score = cosine_similarity(emb, ref_emb)
        if score > best_score:
            best_score = score
            best_name = name

    confidence = (best_score + 1) / 2
    if confidence < THRESHOLD:
        return "unknown", confidence

    return best_name, confidence

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/warmup")
def warmup():
    load_model()
    return {"status": "ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_READY:
        load_model()

    if not embedding_db:
        return {
            "name": "unknown",
            "confidence": 0.0,
            "reason": "empty_database"
        }

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid_image"}

    emb = extract_embedding(img)
    if emb is None:
        return {"error": "no_face_detected"}

    name, conf = predict_identity(emb)

    return {
        "name": name,
        "confidence": round(conf, 4)
    }
