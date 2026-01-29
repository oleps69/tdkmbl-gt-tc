import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from insightface.app import FaceAnalysis

app = FastAPI()

MODEL = None
EMBEDDINGS_DB = None
THRESHOLD = 0.6

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.on_event("startup")
async def load_model():
    global MODEL, EMBEDDINGS_DB
    
    try:
        MODEL = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        MODEL.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")
    
    try:
        with open("face_embeddings.pkl", "rb") as f:
            EMBEDDINGS_DB = pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError("face_embeddings.pkl not found")
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings: {e}")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB")
        img_array = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")
    
    try:
        faces = MODEL.get(img_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {e}")
    
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected")
    
    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    
    face = faces[0]
    query_embedding = face.normed_embedding
    
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Failed to extract embedding")
    
    names = EMBEDDINGS_DB["names"]
    embeddings = EMBEDDINGS_DB["embeddings"]
    
    similarities = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])
    max_idx = np.argmax(similarities)
    max_similarity = similarities[max_idx]
    
    if max_similarity < THRESHOLD:
        return {
            "name": "unknown",
            "confidence": float(max_similarity),
            "threshold": THRESHOLD
        }
    
    return {
        "name": names[max_idx],
        "confidence": float(max_similarity),
        "threshold": THRESHOLD
    }
