from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, os, uuid

app = FastAPI()

# Allow frontend access (Next.js usually runs on http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("model/best.pt")  # Replace with your model path

@app.get("/")
async def root():
    return {"message": "Backend is running!"}


import base64

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Run YOLO
    results = model(image)

    # Convert annotated image (NumPy array) → PIL → bytes → base64
    annotated_array = results[0].plot()  # this is a NumPy array
    annotated_image = Image.fromarray(annotated_array)

    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    # Extract labels & confidences
    detections = []
    for box in results[0].boxes:
        detections.append({
            "label": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()
        })

    # Return JSON with detections and Base64 image
    return JSONResponse({
        "detections": detections,
        "image_base64": img_str
    })


