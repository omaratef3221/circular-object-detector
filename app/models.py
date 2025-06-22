from pydantic import BaseModel
from typing import List, Optional
import uuid

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class CircleProperties(BaseModel):
    centroid_x: int
    centroid_y: int
    radius: int
    bounding_box: BoundingBox

class DetectedCircle(BaseModel):
    id: str
    properties: CircleProperties

class ImageAnalysisResult(BaseModel):
    image_id: str
    circles: List[DetectedCircle]
    mask_path: Optional[str] = None
    original_path: str