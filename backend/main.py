"""
Smart Checkout — FastAPI Backend
Replaces the Gradio app with a full REST API + web UI.

Endpoints:
  POST /detect        — Upload an image, run YOLO, return detections + prices
  GET  /products      — List all products with prices
  PUT  /products/{class_name} — Update a product's price
  GET  /              — Serve the web UI
"""

from __future__ import annotations

import io
import os
import sys
import base64
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

from database import init_db, get_product_by_class, get_all_products, update_product_price

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "my_model", "my_model.pt"),
)
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

# ── Startup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Checkout API", version="1.0.0")

# Serve static files (frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize database on startup
init_db()

# Load YOLO model once at startup
print(f"Loading YOLO model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Set the MODEL_PATH environment variable or place your model correctly.")
    sys.exit(1)

try:
    model = YOLO(MODEL_PATH, task="detect")
    labels = model.names  # e.g. {0: 'bounty', 1: 'galaxy', ...}
    class_name_to_idx = {v: k for k, v in labels.items()}
    print(f"Model loaded. Classes: {labels}")
except Exception as e:
    print(f"ERROR: Failed to load YOLO model: {e}")
    sys.exit(1)


# ── Schemas ──────────────────────────────────────────────────────────────────
class Detection(BaseModel):
    class_name: str
    display_name: str
    confidence: float
    price: float
    currency: str
    bbox: List[float]  # [xmin, ymin, xmax, ymax]


class DetectionResponse(BaseModel):
    detections: List[Detection]
    total_items: int
    total_price: float
    currency: str
    annotated_image: str  # base64-encoded JPEG


class PriceUpdate(BaseModel):
    price: float


# ── Helpers ──────────────────────────────────────────────────────────────────
BBOX_COLORS = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209),
    (178, 182, 133), (88, 159, 106), (96, 202, 231),
    (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184),
]


def annotate_frame(frame: np.ndarray, detections_data: List[Detection]) -> np.ndarray:
    """Draw bounding boxes and price labels on the frame."""
    annotated = frame.copy()
    for det in detections_data:
        xmin, ymin, xmax, ymax = [int(v) for v in det.bbox]
        # Pick a color based on class index
        class_idx = class_name_to_idx.get(det.class_name, 0)
        color = BBOX_COLORS[class_idx % len(BBOX_COLORS)]

        # Draw bounding box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

        # Label with name, confidence, and price
        label = f"{det.display_name} {int(det.confidence * 100)}% - {det.currency} {det.price:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(
            annotated,
            (xmin, label_ymin - label_size[1] - 10),
            (xmin + label_size[0], label_ymin + baseline - 10),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated, label,
            (xmin, label_ymin - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return annotated


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main web UI."""
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(image: UploadFile = File(...)):
    """
    Upload an image → run YOLO detection → return detections with prices.
    """
    # Read and decode the uploaded image
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid image file.")

    # Run YOLO inference
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    detections: List[Detection] = []
    total_price = 0.0
    currency = "AED"

    for i in range(len(boxes)):
        conf = boxes[i].conf.item()
        if conf < CONFIDENCE_THRESHOLD:
            continue

        # Bounding box
        xyxy = boxes[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.tolist()

        # Class info
        class_idx = int(boxes[i].cls.item())
        class_name = labels[class_idx]

        # Price lookup
        product = get_product_by_class(class_name)
        if product:
            display_name = product["display_name"]
            price = product["price"]
            currency = product["currency"]
        else:
            display_name = class_name.title()
            price = 0.0

        total_price += price

        detections.append(Detection(
            class_name=class_name,
            display_name=display_name,
            confidence=round(conf, 3),
            price=price,
            currency=currency,
            bbox=[xmin, ymin, xmax, ymax],
        ))

    # Annotate the image
    annotated = annotate_frame(frame, detections)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_image = base64.b64encode(buffer).decode("utf-8")

    return DetectionResponse(
        detections=detections,
        total_items=len(detections),
        total_price=round(total_price, 2),
        currency=currency,
        annotated_image=b64_image,
    )


class FrameRequest(BaseModel):
    image: str  # base64-encoded JPEG frame


@app.post("/detect_frame", response_model=DetectionResponse)
async def detect_frame(body: FrameRequest):
    """
    Accept a base64-encoded camera frame, run YOLO detection,
    and return detections with an annotated image.
    """
    try:
        img_bytes = base64.b64decode(body.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image frame.")

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    detections: List[Detection] = []
    total_price = 0.0
    currency = "AED"

    for i in range(len(boxes)):
        conf = boxes[i].conf.item()
        if conf < CONFIDENCE_THRESHOLD:
            continue

        xyxy = boxes[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.tolist()

        class_idx = int(boxes[i].cls.item())
        class_name = labels[class_idx]

        product = get_product_by_class(class_name)
        if product:
            display_name = product["display_name"]
            price = product["price"]
            currency = product["currency"]
        else:
            display_name = class_name.title()
            price = 0.0

        total_price += price

        detections.append(Detection(
            class_name=class_name,
            display_name=display_name,
            confidence=round(conf, 3),
            price=price,
            currency=currency,
            bbox=[xmin, ymin, xmax, ymax],
        ))

    annotated = annotate_frame(frame, detections)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_image = base64.b64encode(buffer).decode("utf-8")

    return DetectionResponse(
        detections=detections,
        total_items=len(detections),
        total_price=round(total_price, 2),
        currency=currency,
        annotated_image=b64_image,
    )


@app.get("/products")
async def list_products():
    """List all products with their prices."""
    return get_all_products()


@app.put("/products/{class_name}")
async def update_price(class_name: str, body: PriceUpdate):
    """Update the price of a product."""
    if body.price < 0:
        raise HTTPException(status_code=400, detail="Price must be non-negative.")
    updated = update_product_price(class_name, body.price)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Product '{class_name}' not found.")
    return {"message": f"Price for '{class_name}' updated to {body.price:.2f}"}


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
