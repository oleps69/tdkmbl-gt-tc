import os
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile
from insightface.app import FaceAnalysis
from typing import Dict
from io import BytesIO
from PIL import Image

# =====================
# CONFIG
# =====================
EMBEDDINGS_PATH = "face_embeddings.pkl"
THRESHOLD = float(os.getenv("FACE_THRESHOLD", 0.6))

# =====================
# LOAD MODEL
# =====================
app_face = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app_face.prepare(ctx_id=0, det_size=(640, 640))

# =====================
# LOAD EMBEDDINGS
# =====================
with open(EMBEDDINGS_PATH, "rb") as f:
    DATA = pickle.load(f)

# ÖNEMLİ: DATA formatı önceki kodda pickle.dump({"embeddings": list_of_embs, "labels": list_of_labels})
KNOWN_EMBEDDINGS = np.array(DATA["embeddings"], dtype=np.float32)
KNOWN_LABELS = DATA["labels"]

# =====================
# FASTAPI
# =====================
app = FastAPI(title="Face Recognition API")

# =====================
# HELPERS
# =====================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # normalize-free dot (both expected normalized but safe)
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def identify_face(embedding: np.ndarray) -> Dict:
    best_score = -1.0
    best_label = "unknown"

    for known_emb, label in zip(KNOWN_EMBEDDINGS, KNOWN_LABELS):
        score = cosine_similarity(embedding, known_emb)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score < THRESHOLD:
        return {
            "label": "unknown",
            "confidence": round(best_score, 4)
        }

    return {
        "label": best_label,
        "confidence": round(best_score, 4)
    }


def load_image_from_bytes(contents: bytes) -> np.ndarray:
    image = Image.open(BytesIO(contents)).convert("RGB")
    return np.array(image)

# =====================
# ROUTES
# =====================
@app.get("/")
def health():
    return {
        "status": "ok",
        "threshold": THRESHOLD,
        "known_faces": len(set(KNOWN_LABELS))
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = load_image_from_bytes(contents)
    faces = app_face.get(img)

    if not faces:
        return {
            "success": False,
            "message": "No face detected"
        }

    results = []
    for face in faces:
        # face.embedding zaten numpy array
        emb = face.embedding.astype(np.float32)
        # normalize to be safe
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        result = identify_face(emb)
        results.append(result)

    return {
        "success": True,
        "faces_detected": len(results),
        "results": results
    }
