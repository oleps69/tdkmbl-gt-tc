import os
import io
import threading
import pickle
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
PICKLE_PATH = "face_embeddings.pkl"
THRESHOLD = 0.6
DET_SIZE = (640, 640)

# =========================
# APP
# =========================
app = FastAPI(title="Face Recognition API")

# =========================
# GLOBAL STATE
# =========================
MODEL = None
EMBEDDINGS = None          # np.ndarray (N, 512)
NAMES = []                 # list[str]
MODEL_READY = False
LOCK = threading.Lock()

# =========================
# UTILS
# =========================
def cosine_similarity_matrix(db_embeddings, query_embedding):
    # db_embeddings: (N, 512)
    # query_embedding: (512,)
    return np.dot(db_embeddings, query_embedding)


# =========================
# LOAD MODEL & DB
# =========================
def load_model_and_db():
    global MODEL, EMBEDDINGS, NAMES, MODEL_READY

    if MODEL_READY:
        return

    with LOCK:
        if MODEL_READY:
            return

        print("🔵 Loading InsightFace model...")
        MODEL = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"]
        )
        MODEL.prepare(ctx_id=-1, det_size=DET_SIZE)
        print("✅ Model loaded")

        # ---- Load embeddings ----
        if not os.path.exists(PICKLE_PATH):
            raise RuntimeError("face_embeddings.pkl not found")

        raw = pickle.load(open(PICKLE_PATH, "rb"))
        if not isinstance(raw, dict) or len(raw) == 0:
            raise RuntimeError("Embedding pickle invalid or empty")

        names = []
        embs = []

        for person, emb in raw.items():
            emb = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            emb = emb / norm
            names.append(person)
            embs.append(emb)

        if not embs:
            raise RuntimeError("No valid embeddings loaded")

        NAMES = names
        EMBEDDINGS = np.vstack(embs)

        MODEL_READY = True
        print(f"🟢 Loaded {len(NAMES)} identities")
        print("✅ System ready")


# =========================
# ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/warmup")
def warmup():
    load_model_and_db()
    return {"status": "ready", "identities": len(NAMES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    load_model_and_db()

    if file is None:
        raise HTTPException(400, "File missing")

    # ---- Load image ----
    try:
        img_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(pil_img)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # ---- Detect faces ----
    faces = MODEL.get(img)
    if not faces:
        return {
            "name": "unknown",
            "confidence": 0.0,
            "reason": "no_face"
        }

    # ---- Pick largest face ----
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    query_emb = face.embedding
    if query_emb is None:
        raise HTTPException(500, "Embedding extraction failed")

    query_emb = query_emb.astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)

    # ---- Compare ----
    sims = cosine_similarity_matrix(EMBEDDINGS, query_emb)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < THRESHOLD:
        return {
            "name": "unknown",
            "confidence": best_score,
            "threshold": THRESHOLD
        }

    return {
        "name": NAMES[best_idx],
        "confidence": best_score,
        "threshold": THRESHOLD
    }
