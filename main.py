import cv2
import numpy as np
import pickle
from fastapi import FastAPI, UploadFile, File
from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
EMBEDDING_FILE = "face_embeddings.pkl"
THRESHOLD = 0.60
IMAGE_SIZE = (640, 640)

# =========================
# LOAD EMBEDDINGS
# =========================
with open(EMBEDDING_FILE, "rb") as f:
    EMBEDDING_DB = pickle.load(f)

# =========================
# LOAD MODEL
# =========================
app_face = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app_face.prepare(ctx_id=0, det_size=IMAGE_SIZE)

# =========================
# FASTAPI
# =========================
app = FastAPI(title="Face Recognition API")

# =========================
# UTILS
# =========================
def extract_embedding(img):
    faces = app_face.get(img)
    if len(faces) == 0:
        return None

    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    emb = face.embedding.astype(np.float32)
    return emb / np.linalg.norm(emb)

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def predict_identity(emb):
    best_person = None
    best_score = -1

    for person, ref_emb in EMBEDDING_DB.items():
        score = cosine_similarity(emb, ref_emb)
        if score > best_score:
            best_score = score
            best_person = person

    confidence = (best_score + 1) / 2
    return best_person, confidence

# =========================
# ROUTES
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    emb = extract_embedding(img)
    if emb is None:
        return {"person": "unknown", "confidence": 0.0}

    person, confidence = predict_identity(emb)

    if confidence < THRESHOLD:
        return {
            "person": "unknown",
            "confidence": round(confidence, 3)
        }

    return {
        "person": person,
        "confidence": round(confidence, 3)
    }

@app.get("/")
def health():
    return {"status": "ok", "persons": len(EMBEDDING_DB)}
