import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from insightface.app import FaceAnalysis
import threading
import os

app = FastAPI()

MODEL = None
EMBEDDINGS_DB = None
MODEL_READY = False
MODEL_LOCK = threading.Lock()

THRESHOLD = 0.6


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_everything():
    global MODEL, EMBEDDINGS_DB, MODEL_READY

    with MODEL_LOCK:
        if MODEL_READY:
            return

        print("🔵 Loading InsightFace model...")

        MODEL = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        MODEL.prepare(ctx_id=-1, det_size=(640, 640))

        print("🟢 Model loaded")

        try:
            with open("face_embeddings.pkl", "rb") as f:
                EMBEDDINGS_DB = pickle.load(f)
            print("🟢 Embeddings loaded")
        except Exception as e:
            print("⚠️ Embeddings NOT loaded:", e)
            EMBEDDINGS_DB = {
                "names": [],
                "embeddings": []
            }

        MODEL_READY = True


@app.on_event("startup")
def startup_event():
    # ⚠️ ASYNC YAPMIYORUZ — Railway sever
    load_everything()


@app.get("/health")
def health():
    # Railway healthcheck için
    if not MODEL_READY:
        return {"status": "loading"}
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model loading")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        faces = MODEL.get(img_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {e}")

    if len(faces) == 0:
        return {
            "name": "unknown",
            "confidence": 0.0,
            "reason": "no_face"
        }

    # En büyük yüzü al
    faces = sorted(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        reverse=True
    )

    face = faces[0]
    embedding = face.normed_embedding

    if embedding is None:
        raise HTTPException(status_code=500, detail="Embedding extraction failed")

    names = EMBEDDINGS_DB["names"]
    embeddings = EMBEDDINGS_DB["embeddings"]

    if len(embeddings) == 0:
        return {
            "name": "unknown",
            "confidence": 0.0,
            "reason": "empty_database"
        }

    similarities = np.dot(embeddings, embedding)
    max_idx = int(np.argmax(similarities))
    max_similarity = float(similarities[max_idx])

    if max_similarity < THRESHOLD:
        return {
            "name": "unknown",
            "confidence": max_similarity,
            "threshold": THRESHOLD
        }

    return {
        "name": names[max_idx],
        "confidence": max_similarity,
        "threshold": THRESHOLD
    }
