import os
import json
import uuid
from typing import Optional
from fastapi import UploadFile
import magic
from PIL import Image
import io
from .models import DetectedCircle


class ImageStorage:
    def __init__(self, storage_path: str = "storage"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "originals"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "masks"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "ground_truth"), exist_ok=True)

    def save_uploaded_file(self, file: UploadFile) -> str:
        file_content = file.file.read()
        file_type = magic.from_buffer(file_content, mime=True)
        if not file_type.startswith("image/"):
            raise ValueError("Uploaded file is not an image")

        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1] or ".png"
        filename = f"{file_id}{ext}"
        file_path = os.path.join(self.storage_path, "originals", filename)

        with open(file_path, "wb") as f:
            f.write(file_content)

        return file_id, filename

    def get_image_path(self, image_id: str) -> str:
        for filename in os.listdir(os.path.join(self.storage_path, "originals")):
            if filename.startswith(image_id):
                return os.path.join(self.storage_path, "originals", filename)
        raise FileNotFoundError(f"Image with ID {image_id} not found")

    def save_mask(self, image_id: str, mask_image) -> str:
        filename = f"{image_id}_mask.png"
        file_path = os.path.join(self.storage_path, "masks", filename)
        mask_image.save(file_path)
        return file_path

    def save_result_image(self, image_id: str, result_image) -> str:
        filename = f"{image_id}_result.png"
        file_path = os.path.join(self.storage_path, "results", filename)
        result_image.save(file_path)
        return file_path

    def save_ground_truth(self, image_id: str, circles: list[DetectedCircle]) -> str:
        path = os.path.join(self.storage_path, "ground_truth", f"{image_id}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                {
                    "ground_truth": [
                        {
                            "id": c.id,
                            "properties": {
                                "centroid_x": c.properties.centroid_x,
                                "centroid_y": c.properties.centroid_y,
                                "radius": c.properties.radius,
                                "bounding_box": {
                                    "x": c.properties.bounding_box.x,
                                    "y": c.properties.bounding_box.y,
                                    "width": c.properties.bounding_box.width,
                                    "height": c.properties.bounding_box.height,
                                },
                            },
                        }
                        for c in circles
                    ]
                },
                f,
                indent=2,
            )

        return path