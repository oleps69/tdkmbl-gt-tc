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

KNOWN_EMBEDDINGS = DATA["embeddings"]
KNOWN_LABELS = DATA["labels"]

# =====================
# FASTAPI
# =====================
app = FastAPI(title="Face Recognition API")

# =====================
# HELPERS
# =====================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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


def load_image(file: UploadFile) -> np.ndarray:
    image = Image.open(BytesIO(file.file.read())).convert("RGB")
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
    img = load_image(file)
    faces = app_face.get(img)

    if not faces:
        return {
            "success": False,
            "message": "No face detected"
        }

    results = []
    for face in faces:
        result = identify_face(face.embedding)
        results.append(result)

    return {
        "success": True,
        "faces_detected": len(results),
        "results": results
    }
