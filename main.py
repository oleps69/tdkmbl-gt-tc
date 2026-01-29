import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from insightface.app import FaceAnalysis
import threading
import os

app = FastAPI()

# --------- GLOBAL STATE ----------
MODEL = None
EMBEDDINGS_DB = {"names": [], "embeddings": []}  # default boş
MODEL_READY = False
MODEL_LOCK = threading.Lock()

THRESHOLD = 0.6  # Cosine similarity threshold


# --------- UTILS ----------
def cosine_similarity(a, b):
    return float(np.dot(a, b))


# --------- MODEL & EMBEDDINGS LOADER ----------
def ensure_model_loaded():
    global MODEL, EMBEDDINGS_DB, MODEL_READY

    if MODEL_READY:
        return

    with MODEL_LOCK:
        if MODEL_READY:
            return

        print("🔵 Loading InsightFace model...")

        # Model setup (minimal)
        MODEL = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"]
        )
        MODEL.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ Model loaded")

        # Load embeddings from training pickle
        pickle_path = "face_embeddings.pkl"
        if os.path.exists(pickle_path):
            try:
                raw_data = pickle.load(open(pickle_path, "rb"))
                if isinstance(raw_data, dict):
                    # Convert {person: embedding} -> {"names": [], "embeddings": []}
                    names = []
                    embeddings = []
                    for person, emb in raw_data.items():
                        emb = np.array(emb, dtype=np.float32)
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                            names.append(person)
                            embeddings.append(emb)
                    EMBEDDINGS_DB = {"names": names, "embeddings": embeddings}
                    print(f"🟢 Loaded {len(names)} embeddings from pickle")
                else:
                    raise ValueError("Pickle format invalid, expected dict")
            except Exception as e:
                print("⚠️ Failed to load embeddings:", e)
                EMBEDDINGS_DB = {"names": [], "embeddings": []}
        else:
            print("⚠️ face_embeddings.pkl not found, DB empty")

        MODEL_READY = True
        print("✅ Model ready")


# --------- HEALTH ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------- WARMUP ----------
@app.post("/warmup")
def warmup():
    ensure_model_loaded()
    return {"status": "ready"}


# --------- PREDICT ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ensure_model_loaded()

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Load image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Detect faces
    try:
        faces = MODEL.get(img_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {e}")

    if not faces:
        return {"name": "unknown", "confidence": 0.0, "reason": "no_face"}

    # Pick largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    query_embedding = face.normed_embedding

    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Embedding extraction failed")

    # Check DB
    if not EMBEDDINGS_DB["embeddings"]:
        return {"name": "unknown", "confidence": 0.0, "reason": "empty_database"}

    # Compute similarities
    similarities = np.dot(EMBEDDINGS_DB["embeddings"], query_embedding)
    max_idx = int(np.argmax(similarities))
    max_similarity = float(similarities[max_idx])

    if max_similarity < THRESHOLD:
        return {"name": "unknown", "confidence": max_similarity, "threshold": THRESHOLD}

    return {
        "name": EMBEDDINGS_DB["names"][max_idx],
        "confidence": max_similarity,
        "threshold": THRESHOLD
    }
