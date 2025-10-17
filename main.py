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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Run YOLO
    results = model(image)

    # Save annotated image with unique filename
    output_name = f"{uuid.uuid4()}.jpg"
    save_path = os.path.join("static", output_name)
    results[0].save(filename=save_path)

    # Extract labels & confidences
    detections = []
    for box in results[0].boxes:
        detections.append({
            "label": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()
        })

    return JSONResponse({
        "detections": detections,
        "image_url": f"/static/{output_name}"
    })
