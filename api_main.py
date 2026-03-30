import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from tensorflow.keras.models import load_model


MODEL_PATH = os.getenv("MODEL_PATH", "dress_code_detector (6).h5")
THRESHOLD = float(os.getenv("DRESS_THRESHOLD", "0.5"))

app = FastAPI(title="Dress Code Detector API", version="1.0.0")

model = None
img_h = 224
img_w = 224


def resolve_model_path() -> str:
    # Prefer explicit MODEL_PATH, then common filenames, then any .h5 in project root.
    candidates = [
        Path(MODEL_PATH),
        Path("dress_code_detector (6).h5"),
        Path("dress_code_detector.h5"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    any_h5 = sorted(Path(".").glob("*.h5"))
    if any_h5:
        return str(any_h5[0])

    raise RuntimeError(
        "Model file not found. Set MODEL_PATH env var or include a .h5 model file in the app root."
    )


@app.on_event("startup")
def load_detection_model() -> None:
    global model, img_h, img_w

    model_file = resolve_model_path()
    model = load_model(model_file)
    _, img_h, img_w, _ = model.input_shape


def classify_image(bgr_image: np.ndarray) -> Dict[str, Any]:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    sized = cv2.resize(rgb, (img_w, img_h))
    inp = sized.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)

    prediction = model.predict(inp, verbose=0)
    score = float(prediction[0][0])
    is_non_compliant = score > THRESHOLD
    result = "NON-COMPLIANT" if is_non_compliant else "COMPLIANT"
    confidence = abs(score - THRESHOLD) * (1.0 / max(THRESHOLD, 1.0 - THRESHOLD))
    confidence = float(max(0.0, min(confidence, 1.0)))

    return {
        "result": result,
        "score": round(score, 6),
        "confidence": round(confidence, 6),
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_path": resolve_model_path() if model is not None else MODEL_PATH,
        "input_size": [img_w, img_h],
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> Dict[str, Any]:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return classify_image(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")