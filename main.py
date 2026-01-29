"""
Memory-Optimized Face Recognition REST API
CPU-only, stable InsightFace setup
No FaceAnalysis, only detector + recognizer
"""

import os
import pickle
import threading
import gc
from typing import Dict
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image

# ===================== GLOBALS =====================

app = FastAPI(title="Face Recognition API (Stable)", version="2.1.0")

face_detector = None
face_recognizer = None
face_embeddings_db = None
model_lock = threading.Lock()

EMBEDDING_DB_PATH = "face_embeddings.pkl"
RECOGNITION_THRESHOLD = 0.60
DET_THRESH = 0.5
INPUT_SIZE = (112, 112)

# ===================== MODEL LOADING =====================

def load_lightweight_models():
    global face_detector, face_recognizer

    if face_detector is not None and face_recognizer is not None:
        return face_detector, face_recognizer

    with model_lock:
        if face_detector is not None and face_recognizer is not None:
            return face_detector, face_recognizer

        try:
            from insightface.model_zoo import model_zoo

            # ---- Detection (CPU friendly & stable) ----
            print("Loading face detector...")
            face_detector = model_zoo.get_model("retinaface_mnet025_v2")
            if face_detector is None:
                raise RuntimeError("retinaface_mnet025_v2 could not be loaded")

            face_detector.prepare(ctx_id=-1, input_size=(320, 320))

            # ---- Recognition (MobileFaceNet) ----
            print("Loading face recognizer...")
            face_recognizer = model_zoo.get_model("arcface_mobileface_v1")
            if face_recognizer is None:
                raise RuntimeError("arcface_mobileface_v1 could not be loaded")

            face_recognizer.prepare(ctx_id=-1)

            gc.collect()
            print("Models loaded successfully")
            return face_detector, face_recognizer

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

# ===================== EMBEDDINGS =====================

def load_embedding_database() -> Dict[str, np.ndarray]:
    global face_embeddings_db

    if face_embeddings_db is not None:
        return face_embeddings_db

    with model_lock:
        if face_embeddings_db is not None:
            return face_embeddings_db

        if not os.path.exists(EMBEDDING_DB_PATH):
            face_embeddings_db = {}
            return face_embeddings_db

        try:
            with open(EMBEDDING_DB_PATH, "rb") as f:
                db = pickle.load(f)

            normalized = {}
            for name, emb in db.items():
                emb = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                normalized[name] = emb

            face_embeddings_db = normalized
            return face_embeddings_db

        except Exception as e:
            raise RuntimeError(f"Embedding DB load failed: {e}")

# ===================== UTILS =====================

def get_largest_face(bboxes):
    if bboxes is None or len(bboxes) == 0:
        return None

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    return int(np.argmax(areas))


def align_face(img, landmark):
    from skimage import transform as trans

    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)

    aligned = trans.warp(img, tform.inverse, output_shape=(112, 112))
    return (aligned * 255).astype(np.uint8)


def recognize_face(embedding, db):
    if not db:
        return "unknown", 0.0

    embedding = embedding / np.linalg.norm(embedding)
    best_name, best_score = "unknown", 0.0

    for name, db_emb in db.items():
        score = float(np.dot(embedding, db_emb))
        if score > best_score:
            best_name, best_score = name, score

    return best_name, best_score

# ===================== API =====================

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/warmup")
def warmup():
    try:
        d, r = load_lightweight_models()
        db = load_embedding_database()
        return {
            "status": "ready",
            "detector": d is not None,
            "recognizer": r is not None,
            "embeddings": len(db)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        img = Image.open(BytesIO(await image.read())).convert("RGB")
        img = np.array(img)[:, :, ::-1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        detector, recognizer = load_lightweight_models()
        db = load_embedding_database()

        bboxes, landmarks = detector.detect(img, threshold=DET_THRESH)
        if bboxes is None or len(bboxes) == 0:
            return {"name": "unknown", "confidence": 0.0, "reason": "no_face"}

        idx = get_largest_face(bboxes)
        face = align_face(img, landmarks[idx])

        emb = recognizer.get_embedding(face)
        name, score = recognize_face(emb, db)

        if score < RECOGNITION_THRESHOLD:
            return {"name": "unknown", "confidence": round(score, 4)}

        return {
            "name": name,
            "confidence": round(score, 4),
            "threshold": RECOGNITION_THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        gc.collect()

# ===================== RUN =====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
