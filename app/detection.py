from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple
from .models import CircleProperties, DetectedCircle, BoundingBox
import filetype

class CircleDetector:
    def __init__(self, min_area: int = 1000):
        self.min_area = min_area  # ignore small blobs

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return edges

    def detect_circles(self, image_path: str) -> Tuple[List[DetectedCircle], Image.Image, Image.Image]:
        if not filetype.is_image(image_path):
            raise ValueError("Provided path does not point to a valid image file.")
        
        img = cv2.imread(image_path)
        edges = self.preprocess(img)
        circles = cv2.HoughCircles(
            image=edges,
            method=cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=40,
            param1=100,
            param2=30,
            minRadius=20,
            maxRadius=140
        )

        h, w = img.shape[:2] # skip the RGB count channel (last one)
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        annotated = img.copy()
        detected: List[DetectedCircle] = []
        circles = np.uint16(np.around(circles))
        # Annotate
        for i, (x, y, r) in enumerate(circles[0, :]):
            cv2.circle(mask, (x, y), r, (0, 255,0), thickness= 4) #circle mask
            cv2.circle(annotated, (x, y), r, (0, 255, 0), thickness=4) # circle image
            cv2.putText(annotated, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)

            props = CircleProperties(
                centroid_x=x,
                centroid_y=y,
                radius=r,
                bounding_box=BoundingBox(x=x - r, y=y - r, width=2*r, height=2*r)
            )
            detected.append(DetectedCircle(id=str(i + 1), properties=props))

        mask_img = Image.fromarray(mask)
        result_img = Image.fromarray(annotated)

        return detected, mask_img, result_img

    def evaluate_detection(self, detected: List[DetectedCircle], ground_truth: List[DetectedCircle], iou_thresh: float = 25.0,) -> dict:
        matched_gt, matched_det = set(), set()

        for d in detected:
            dx, dy = d.properties.centroid_x, d.properties.centroid_y
            for g in ground_truth:
                gx, gy = g.properties.centroid_x, g.properties.centroid_y
                if np.hypot(dx - gx, dy - gy) <= iou_thresh:
                    matched_gt.add(g.id)
                    matched_det.add(d.id)
                    break

        tp = len(matched_det)
        fp = len(detected) - tp
        fn = len(ground_truth) - len(matched_gt)

        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

        return {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
