from __future__ import annotations
import numpy as np
from utils.normalization import clamp


def extract_objects(detections: list[dict], image_shape: tuple[int, ...]) -> dict:
    height, width = image_shape[:2]
    image_area = height * width
    people = [d for d in detections if d["label"] == "person"]
    areas = [max(0., (d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1])) / image_area for d in people]
    labels = {d["label"] for d in detections}
    person_area_ratio = min(1., sum(areas))
    return {"person_count": len(people), "person_area_ratio": person_area_ratio,
            "largest_person_area": max(areas, default=0.), "microphone_present": int("microphone" in labels),
            "laptop_present": int("laptop" in labels), "podium_present": int("podium" in labels), "people": people}


def event_coverage_score(objects: dict) -> float:
    count_score = min(100., objects["person_count"] / 15 * 100)
    area_score = min(100., objects["person_area_ratio"] / .50 * 100)
    context_score = 100. if (objects["microphone_present"] or objects["laptop_present"] or objects["podium_present"]) else 45.
    return clamp(.45 * count_score + .40 * area_score + .15 * context_score)
