"""
Memory-Optimized Face Recognition REST API
CPU-only, minimal memory footprint, Docker-compatible
Uses lightweight model with manual memory management
"""

import os
import pickle
import threading
import gc
from typing import Optional, Dict, Any
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image

# Global variables
app = FastAPI(title="Face Recognition API (Optimized)", version="2.0.0")
face_detector = None
face_recognizer = None
face_embeddings_db = None
model_lock = threading.Lock()

# Configuration
EMBEDDING_DB_PATH = "face_embeddings.pkl"
RECOGNITION_THRESHOLD = 0.35  # Slightly lower for lighter models
DET_THRESH = 0.5  # Detection confidence threshold
INPUT_SIZE = (112, 112)  # Standard face recognition input size


def load_lightweight_models():
    """
    Load minimal InsightFace models separately to reduce memory.
    Uses only detection and recognition, skipping other analysis modules.
    """
    global face_detector, face_recognizer
    
    if face_detector is not None and face_recognizer is not None:
        return face_detector, face_recognizer
    
    with model_lock:
        # Double-check pattern
        if face_detector is not None and face_recognizer is not None:
            return face_detector, face_recognizer
        
        try:
            # Import only when needed
            from insightface.model_zoo import model_zoo
            
            # Load only detection model (much lighter than full FaceAnalysis)
            print("Loading detection model...")
            face_detector = model_zoo.get_model('retinaface_r50_v1')
            face_detector.prepare(ctx_id=-1, input_size=(640, 640))
            
            # Load only recognition model
            print("Loading recognition model...")
            face_recognizer = model_zoo.get_model('arcface_r100_v1')
            face_recognizer.prepare(ctx_id=-1)
            
            # Force garbage collection
            gc.collect()
            
            print("Models loaded successfully!")
            return face_detector, face_recognizer
            
        except Exception as e:
            # Fallback to even lighter model
            try:
                print(f"Primary models failed: {e}")
                print("Trying lighter model combination...")
                
                from insightface.model_zoo import model_zoo
                
                # Use smaller detection model
                face_detector = model_zoo.get_model('retinaface_mnet025_v2')
                face_detector.prepare(ctx_id=-1, input_size=(320, 320))
                
                # Use smaller recognition model  
                face_recognizer = model_zoo.get_model('arcface_mobileface_v1')
                face_recognizer.prepare(ctx_id=-1)
                
                gc.collect()
                
                print("Lightweight models loaded successfully!")
                return face_detector, face_recognizer
                
            except Exception as e2:
                raise RuntimeError(f"Failed to load any models: {str(e2)}")


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


def get_largest_face(bboxes, landmarks):
    """
    Select the largest face from detected faces based on bounding box area.
    Returns the index of the largest face.
    """
    if bboxes is None or len(bboxes) == 0:
        return None
    
    max_area = 0
    max_idx = 0
    
    for idx, bbox in enumerate(bboxes):
        # bbox format: [x1, y1, x2, y2, score]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        if area > max_area:
            max_area = area
            max_idx = idx
    
    return max_idx


def align_face(img, landmark):
    """
    Simple face alignment using landmarks.
    Returns aligned face for better recognition.
    """
    from skimage import transform as trans
    
    # Standard face template (5 landmarks)
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
    aligned = (aligned * 255).astype(np.uint8)
    
    return aligned


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
    return {"status": "healthy", "service": "face-recognition-api-optimized"}


@app.post("/warmup")
async def warmup():
    """
    Pre-load models and embedding database.
    Call this endpoint after deployment to avoid cold start on first prediction.
    """
    try:
        detector, recognizer = load_lightweight_models()
        embeddings_db = load_embedding_database()
        
        return {
            "status": "success",
            "detector_loaded": detector is not None,
            "recognizer_loaded": recognizer is not None,
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
        
        # Convert PIL image to numpy array (RGB format)
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for InsightFace (it expects BGR)
        image_bgr = image_array[:, :, ::-1]
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )
    
    try:
        # Load models and database
        detector, recognizer = load_lightweight_models()
        embeddings_db = load_embedding_database()
        
        # Check if database is empty
        if not embeddings_db:
            return JSONResponse({
                "name": "unknown",
                "confidence": 0.0,
                "reason": "empty_database"
            })
        
        # Detect faces
        bboxes, landmarks = detector.detect(image_bgr, threshold=DET_THRESH)
        
        # Check if any face detected
        if bboxes is None or len(bboxes) == 0:
            return JSONResponse({
                "name": "unknown",
                "confidence": 0.0,
                "reason": "no_face"
            })
        
        # Select largest face
        face_idx = get_largest_face(bboxes, landmarks)
        
        if face_idx is None:
            return JSONResponse({
                "name": "unknown",
                "confidence": 0.0,
                "reason": "no_face"
            })
        
        # Get landmark for alignment
        landmark = landmarks[face_idx]
        
        # Align face
        try:
            aligned_face = align_face(image_bgr, landmark)
        except:
            # Fallback: crop face without alignment
            bbox = bboxes[face_idx]
            x1, y1, x2, y2 = map(int, bbox[:4])
            face_crop = image_bgr[y1:y2, x1:x2]
            
            # Resize to standard size
            from PIL import Image as PILImage
            face_pil = PILImage.fromarray(face_crop[:, :, ::-1])
            face_pil = face_pil.resize(INPUT_SIZE)
            aligned_face = np.array(face_pil)[:, :, ::-1]
        
        # Get face embedding
        face_embedding = recognizer.get_embedding(aligned_face)
        
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
    finally:
        # Force garbage collection after each request to free memory
        gc.collect()


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
