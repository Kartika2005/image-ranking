from __future__ import annotations
import numpy as np
from utils.normalization import clamp


def extract_composition(objects: dict, image_shape: tuple[int, ...]) -> dict:
    height, width = image_shape[:2]
    people = objects["people"]
    if not people:
        return {"composition_score": 35., "edge_crowding": 0., "central_subject_score": 30., "empty_space_score": 45.}
    centers = np.array([((p["xyxy"][0] + p["xyxy"][2]) / 2 / width, (p["xyxy"][1] + p["xyxy"][3]) / 2 / height) for p in people])
    areas = np.array([max(0., (p["xyxy"][2]-p["xyxy"][0])*(p["xyxy"][3]-p["xyxy"][1])) / (height*width) for p in people])
    weighted_center_x = float(np.average(centers[:, 0], weights=np.maximum(areas, .001)))
    central_subject_score = clamp(100 - abs(weighted_center_x - .5) * 130)
    edge_crowding = float(((centers[:, 0] < .07) | (centers[:, 0] > .93)).mean())
    edge_score = 100 * (1 - edge_crowding)
    empty_ratio = max(0., 1 - objects["person_area_ratio"])
    empty_space_score = clamp(100 - abs(empty_ratio - .65) * 120)
    return {"composition_score": clamp(.45 * central_subject_score + .30 * edge_score + .25 * empty_space_score),
            "edge_crowding": edge_crowding, "central_subject_score": central_subject_score, "empty_space_score": empty_space_score}
