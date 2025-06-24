from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple
from .models import CircleProperties, DetectedCircle, BoundingBox
import filetype
import json
import os

class CircleDetector:
    def __init__(self, min_area: int = 1000):
        self.min_area = min_area  # ignore small blobs

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        blurred = cv2.medianBlur(gray, 7)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return edges

    def detect_circles(self, image_path: str) -> Tuple[List[DetectedCircle], Image.Image, Image.Image]:
        if not filetype.is_image(image_path):
            raise ValueError("Provided path does not point to a valid image file.")
        
        img = cv2.imread(image_path)
        edges = self.preprocess(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            image=edges,
            method=cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=100,
            param2=25,
            minRadius=20,
            maxRadius=120
        )

        if circles is None:
            return [], Image.fromarray(edges), Image.fromarray(img)

        h, w = img.shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        annotated = img.copy()
        detected: List[DetectedCircle] = []

        circles = np.uint16(np.around(circles))

        for i, (x, y, r) in enumerate(circles[0, :]):
            circle_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(circle_mask, (x, y), r, 255, thickness=-1)

            area = cv2.countNonZero(circle_mask)
            ideal_area = np.pi * (r ** 2)
            roundness = area / ideal_area if ideal_area != 0 else 0

            if roundness < 0.85 or roundness > 1.15:
                continue

            cv2.circle(mask, (x, y), r, (0, 255, 0), thickness=4)
            cv2.circle(annotated, (x, y), r, (0, 255, 0), thickness=4)
            cv2.putText(annotated, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)

            props = CircleProperties(
                centroid_x=x,
                centroid_y=y,
                radius=r,
                bounding_box=BoundingBox(x=x - r, y=y - r, width=2 * r, height=2 * r)
            )
            detected.append(DetectedCircle(id=str(i + 1), properties=props))

        mask_img = Image.fromarray(mask)
        result_img = Image.fromarray(annotated)

        return detected, mask_img, result_img

    def evaluate_detection(self, image_filename: str, coco_json_path: str, storage, DetectedCircle, CircleProperties, BoundingBox, iou_thresh: float = 25.0) -> dict:
        with open(coco_json_path, "r") as f:
            coco = json.load(f)
        image_info = next((img for img in coco["images"] if img["file_name"] == image_filename), None)
        if not image_info:
            raise ValueError(f"{image_filename} not found in COCO JSON")

        gt_path = os.path.join(storage.storage_path, "ground_truth", f"{image_filename}.json")
        if not os.path.exists(gt_path):
            raise ValueError(f"Ground truth (detected result) not found for {image_filename}")
        
        with open(gt_path, "r") as f:
            detected_data = json.load(f)

        detected = []
        for c in detected_data["ground_truth"]:
            props = c["properties"]
            bbox = props["bounding_box"]
            detected.append(
                DetectedCircle(
                    id=c["id"],
                    properties=CircleProperties(
                        centroid_x=props["centroid_x"],
                        centroid_y=props["centroid_y"],
                        radius=props["radius"],
                        bounding_box=BoundingBox(
                            x=bbox["x"], y=bbox["y"], width=bbox["width"], height=bbox["height"]
                        ),
                    )
                )
            )

        # load actual ground truth from coco
        annotations = [a for a in coco["annotations"] if a["image_id"] == image_info["id"]]
        ground_truth = []
        for i, ann in enumerate(annotations, 1):
            x, y, w, h = ann["bbox"]
            ground_truth.append(
                DetectedCircle(
                    id=str(i),
                    properties=CircleProperties(
                        centroid_x=int(x + w / 2),
                        centroid_y=int(y + h / 2),
                        radius=int(min(w, h) / 2),
                        bounding_box=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h))
                    )
                )
            )

        matched_gt_ids = set()
        matched_det_ids = set()

        for i, d in enumerate(detected):
            dx, dy = d.properties.centroid_x, d.properties.centroid_y
            for j, g in enumerate(ground_truth):
                gx, gy = g.properties.centroid_x, g.properties.centroid_y
                if j not in matched_gt_ids and np.hypot(dx - gx, dy - gy) <= iou_thresh:
                    matched_gt_ids.add(j)
                    matched_det_ids.add(i)
                    break

        correct = len(matched_gt_ids)
        total_gt = len(ground_truth)
        total_det = len(detected)

        missed = total_gt - correct
        false_positives = total_det - correct

        accuracy = correct / total_gt if total_gt else 0

        return {
            "accuracy": round(accuracy, 3),
            "coins_detected": total_det,
            "ground_truth_total": total_gt,
            "correct_matches": correct,
            "false_positives": false_positives,
            "missed": missed,
            "summary": f"{correct} of {total_gt} coins detected — {false_positives} false positives, {missed} missed"
        }