from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import os
import traceback
from .models import ImageAnalysisResult, DetectedCircle, CircleProperties
from .storage import ImageStorage
from .detection import CircleDetector
import json

app = FastAPI()
storage = ImageStorage() # Initialize storage object
detector = CircleDetector() # Initialize circle detector object

# In-memory storage for analysis results 
# (in production, use a database e.g. AWS S3)
analysis_results = {}

@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    try:
        image_id, _ = storage.save_uploaded_file(image)
        image_path = storage.get_image_path(image_id)

        circles, mask, result_image = detector.detect_circles(image_path)

        mask_path = storage.save_mask(image_id, mask)
        result_path = storage.save_result_image(image_id, result_image)

        analysis_results[image_id] = ImageAnalysisResult(
            image_id=image_id,
            circles=circles,
            mask_path=mask_path,
            original_path=image_path
        )
        gt_path = storage.save_ground_truth(image_id, circles)
        return {"image_id": image_id, "detected_circles": len(circles)}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/list-circles/{image_id}")
async def list_circles(image_id: str):
    print(f"Listing circles for image_id: {image_id}")
    print(f"Current analysis results: {analysis_results.keys()}")
    if image_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Image not found")
    
    result = analysis_results[image_id]
    print(f"Found result for image_id: {image_id}, circles: {len(result.circles)}")
    return {
        "image_id": image_id,
        "circles": [
            {"id": circle.id, "bounding_box": circle.properties.bounding_box}
            for circle in result.circles
        ]
    }

@app.get("/circle-details/{image_id}/{circle_id}")
async def circle_details(image_id: str, circle_id: str):
    if image_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Image not found")
    
    result = analysis_results[image_id]
    circle = next((c for c in result.circles if c.id == circle_id), None)
    
    if not circle:
        raise HTTPException(status_code=404, detail="Circle not found")
    
    return {
        "image_id": image_id,
        "circle_id": circle_id,
        "centroid": (circle.properties.centroid_x, circle.properties.centroid_y),
        "radius": circle.properties.radius,
        "bounding_box": circle.properties.bounding_box
    }

@app.get("/view-result/{image_id}")
async def view_result(image_id: str):
    if image_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Image not found")
    
    result = analysis_results[image_id]
    result_path = os.path.join(storage.storage_path, "results", f"{image_id}_result.png")
    
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result image not found")
    
    return FileResponse(result_path)

@app.get("/evaluate-auto/{image_id}")
async def evaluate_auto(image_id: str):
    if image_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Image not found")
    
    gt_path = os.path.join(storage.storage_path, "ground_truth", f"{image_id}.json")
    if not os.path.exists(gt_path):
        raise HTTPException(status_code=404, detail="Ground truth file not found")

    with open(gt_path, "r") as f:
        data = json.load(f)

    ground_truth = [DetectedCircle(**circle) for circle in data["ground_truth"]]
    detected = analysis_results[image_id].circles
    return detector.evaluate_detection(detected, ground_truth)