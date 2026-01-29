import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from insightface.app import FaceAnalysis
import threading

app = FastAPI()

MODEL = None
EMBEDDINGS_DB = None
MODEL_READY = False
MODEL_LOCK = threading.Lock()

THRESHOLD = 0.6


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def ensure_model_loaded():
    global MODEL, EMBEDDINGS_DB, MODEL_READY

    if MODEL_READY:
        return

    with MODEL_LOCK:
        if MODEL_READY:
            return

        print("🔵 Lazy-loading InsightFace model...")

        MODEL = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        MODEL.prepare(ctx_id=-1, det_size=(640, 640))

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
        print("✅ Model ready")


@app.get("/health")
def health():
    # Railway SADECE BUNU İSTİYOR
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 🔑 MODEL BURADA YÜKLENİR
    ensure_model_loaded()

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    faces = MODEL.get(img_array)

    if len(faces) == 0:
        return {
            "name": "unknown",
            "confidence": 0.0,
            "reason": "no_face"
        }

    faces = sorted(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        reverse=True
    )

    face = faces[0]
    embedding = face.normed_embedding

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
