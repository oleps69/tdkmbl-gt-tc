"""
Production-Ready Face Recognition REST API using InsightFace (buffalo_l)
CPU-only, Docker-compatible, headless environment ready
"""

import os
import pickle
import threading
from typing import Optional, Dict, Any
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image

# Global variables
app = FastAPI(title="Face Recognition API", version="1.0.0")
face_model = None
face_embeddings_db = None
model_lock = threading.Lock()

# Configuration
EMBEDDING_DB_PATH = "face_embeddings.pkl"
RECOGNITION_THRESHOLD = 0.40  # Cosine similarity threshold
DET_SIZE = (640, 640)
MODEL_NAME = "buffalo_l"


def load_insightface_model():
    """
    Lazy load InsightFace model with CPU configuration.
    This function is called only when needed, not at import time.
    """
    global face_model
    
    if face_model is not None:
        return face_model
    
    with model_lock:
        # Double-check pattern
        if face_model is not None:
            return face_model
        
        try:
            # Import here to avoid crash at startup
            from insightface.app import FaceAnalysis
            
            # Initialize with CPU-only configuration
            model = FaceAnalysis(
                name=MODEL_NAME,
                providers=['CPUExecutionProvider']
            )
            
            # Prepare with CPU context and detection size
            model.prepare(
                ctx_id=-1,  # CPU
                det_size=DET_SIZE
            )
            
            face_model = model
            return face_model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load InsightFace model: {str(e)}")


def load_embedding_database() -> Dict[str, np.ndarray]:
    """
    Load pre-trained face embeddings from pickle file.
    All embeddings are normalized to unit vectors (L2 norm).
    """
    global face_embeddings_db
    
    if face_embeddings_db is not None:
        return face_embeddings_db
    
    with model_lock:
        # Double-check pattern
        if face_embeddings_db is not None:
            return face_embeddings_db
        
        if not os.path.exists(EMBEDDING_DB_PATH):
            # Empty database is valid - return empty dict
            face_embeddings_db = {}
            return face_embeddings_db
        
        try:
            with open(EMBEDDING_DB_PATH, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Ensure all embeddings are float32 and normalized
            normalized_embeddings = {}
            for name, embedding in embeddings.items():
                embedding = np.array(embedding, dtype=np.float32)
                
                # L2 normalization
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                normalized_embeddings[name] = embedding
            
            face_embeddings_db = normalized_embeddings
            return face_embeddings_db
            
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding database: {str(e)}")


def get_largest_face(faces):
    """
    Select the largest face from detected faces based on bounding box area.
    """
    if not faces:
        return None
    
    largest_face = None
    max_area = 0
    
    for face in faces:
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        if area > max_area:
            max_area = area
            largest_face = face
    
    return largest_face


def recognize_face(face_embedding: np.ndarray, embeddings_db: Dict[str, np.ndarray]) -> tuple:
    """
    Recognize face using cosine similarity.
    Since embeddings are normalized, we can use dot product.
    
    Returns:
        (name, confidence_score)
    """
    if not embeddings_db:
        return "unknown", 0.0
    
    # Normalize query embedding
    norm = np.linalg.norm(face_embedding)
    if norm > 0:
        face_embedding = face_embedding / norm
    
    # Calculate cosine similarity with all database embeddings
    best_name = "unknown"
    best_score = 0.0
    
    for name, db_embedding in embeddings_db.items():
        # Dot product of normalized vectors = cosine similarity
        similarity = np.dot(face_embedding, db_embedding)
        
        if similarity > best_score:
            best_score = similarity
            best_name = name
    
    return best_name, float(best_score)


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy", "service": "face-recognition-api"}


@app.post("/warmup")
async def warmup():
    """
    Pre-load model and embedding database.
    Call this endpoint after deployment to avoid cold start on first prediction.
    """
    try:
        load_insightface_model()
        embeddings_db = load_embedding_database()
        
        return {
            "status": "success",
            "model_loaded": face_model is not None,
            "embeddings_count": len(embeddings_db)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Face recognition endpoint.
    Accepts a single image file and returns the recognized person's name.
    """
    # Validate file upload
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image."
        )
    
    try:
        # Read image bytes
        image_bytes = await image.read()
        
        # Load image using PIL (headless-compatible)
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL image to numpy array
        image_array = np.array(pil_image)
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )
    
    try:
        # Load model and database
        model = load_insightface_model()
        embeddings_db = load_embedding_database()
        
        # Check if database is empty
        if not embeddings_db:
            return JSONResponse({
                "name": "unknown",
                "confidence": 0.0,
                "reason": "empty_database"
            })
        
        # Detect faces
        faces = model.get(image_array)
        
        # Check if any face detected
        if not faces:
            return JSONResponse({
                "name": "unknown",
                "confidence": 0.0,
                "reason": "no_face"
            })
        
        # Select largest face
        largest_face = get_largest_face(faces)
        
        if largest_face is None:
            return JSONResponse({
                "name": "unknown",
                "confidence": 0.0,
                "reason": "no_face"
            })
        
        # Get face embedding
        face_embedding = largest_face.normed_embedding
        
        # Recognize face
        recognized_name, confidence = recognize_face(face_embedding, embeddings_db)
        
        # Check threshold
        if confidence < RECOGNITION_THRESHOLD:
            return JSONResponse({
                "name": "unknown",
                "confidence": round(confidence, 4),
                "threshold": RECOGNITION_THRESHOLD
            })
        
        # Success
        return JSONResponse({
            "name": recognized_name,
            "confidence": round(confidence, 4),
            "threshold": RECOGNITION_THRESHOLD
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
