# Circular Object (Coin) Detector

A FastAPI-based web application for detecting coins (circular objects) in photos. Upload an image, and the app will automatically detect coins, generate masks, annotate results, and provide detailed information about each detected coin.

## Features
- **Automatic coin detection** using OpenCV's Hough Circle Transform
- **REST API** for uploading images, listing detected coins, viewing details, and evaluating results
- **Result visualization**: annotated images and binary masks
- **Ground truth storage** for evaluation
- **Docker support** for easy deployment

## How It Works
1. **Upload a photo** containing coins.
2. The app detects circles (coins) using image processing and Hough Circle Transform.
3. It generates:
   - An annotated result image with detected coins highlighted
   - A binary mask image
   - A JSON file with detected coin properties (centroid, radius, bounding box)
4. You can retrieve details, masks, and results via the API.

## API Endpoints
- `POST /upload-image` — Upload an image for coin detection
- `GET /list-circles/{image_id}` — List detected coins for an image
- `GET /circle-details/{image_id}/{circle_id}` — Get details for a specific coin
- `GET /view-result/{image_id}` — Download the annotated result image
- `GET /evaluate-auto/{image_id}` — Evaluate detection against ground truth

## Installation
### Requirements
- Python 3.9+
- pip
- [Docker](https://www.docker.com/) (optional)

### Install with pip
```bash
pip install -r requirements.txt
```

### Run the app
```bash
uvicorn app.main:app --reload
```

### Or use Docker
```bash
docker build -t coin-detector .
docker run -p 8000:8000 -v $(pwd)/storage:/app/storage coin-detector
```

## Usage Example
### Upload an image
```bash
curl -F "image=@storage/originals/0d241f6c-bece-402d-a48f-32c810bb2ff0.jpg" http://localhost:8000/upload-image
```
Response:
```json
{"image_id": "0d241f6c-bece-402d-a48f-32c810bb2ff0", "detected_circles": 3}
```

### List detected coins
```bash
curl http://localhost:8000/list-circles/0d241f6c-bece-402d-a48f-32c810bb2ff0
```

### Get details for a coin
```bash
curl http://localhost:8000/circle-details/0d241f6c-bece-402d-a48f-32c810bb2ff0/1
```

### Download result image
```bash
curl -O http://localhost:8000/view-result/0d241f6c-bece-402d-a48f-32c810bb2ff0
```

### Evaluate detection
```bash
curl http://localhost:8000/evaluate-auto/0d241f6c-bece-402d-a48f-32c810bb2ff0
```

## Storage Structure
- `storage/originals/` — Uploaded images
- `storage/results/` — Annotated result images
- `storage/masks/` — Binary mask images
- `storage/ground_truth/` — Ground truth JSON files

## Detection Algorithm
- Converts image to grayscale, applies median blur, and detects edges
- Uses Hough Circle Transform to find circles
- Merges overlapping detections
- Annotates and stores results

## Dependencies
- fastapi
- uvicorn
- opencv-python
- numpy
- Pillow
- python-magic
- pydantic
- scikit-image

## Example Images
- Sample input: `storage/originals/0d241f6c-bece-402d-a48f-32c810bb2ff0.jpg`
- Result: `storage/results/0d241f6c-bece-402d-a48f-32c810bb2ff0_result.png`
- Mask: `storage/masks/0d241f6c-bece-402d-a48f-32c810bb2ff0_mask.png`

---
