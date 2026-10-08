"""Local YOLO wrapper. COCO classes are used only when the model detects them."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np


class YoloModel:
    def __init__(self, model_name: str, confidence: float = 0.25):
        # Keep Ultralytics' writable settings/cache with the project. This also avoids
        # relying on a user-profile directory in restricted or containerized runs.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics to use YOLO.") from exc
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> list[dict]:
        result = self.model(image, conf=self.confidence, verbose=False)[0]
        names = result.names
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({"label": names[class_id], "xyxy": box.xyxy[0].cpu().numpy().astype(float), "confidence": float(box.conf.item())})
        return detections
