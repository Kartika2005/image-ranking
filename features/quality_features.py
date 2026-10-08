from __future__ import annotations
import cv2
import numpy as np
from utils.normalization import bell, clamp, linear


def extract_quality(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    dark_ratio = float((gray < 30).mean())
    bright_ratio = float((gray > 225).mean())
    sharpness_score = linear(sharpness, 20, 450)
    brightness_score = bell(brightness, 135, 58)
    contrast_score = bell(contrast, 55, 28)
    exposure_score = clamp(100 - 150 * max(0, dark_ratio - .08) - 150 * max(0, bright_ratio - .08))
    quality_score = .40 * sharpness_score + .25 * brightness_score + .20 * contrast_score + .15 * exposure_score
    tilt_degrees, tilt_confidence = _estimate_tilt(gray)
    return {"sharpness": sharpness, "brightness": brightness, "contrast": contrast, "dark_ratio": dark_ratio,
            "bright_ratio": bright_ratio, "sharpness_score": sharpness_score, "brightness_score": brightness_score,
            "tilt_degrees": tilt_degrees, "tilt_confidence": tilt_confidence, "quality_score": clamp(quality_score)}


def _estimate_tilt(gray: np.ndarray) -> tuple[float, float]:
    """Estimate roll from long horizontal/vertical room lines.

    Returns a neutral zero angle when there are not enough reliable lines; this avoids
    rejecting close-ups or dark scenes simply because a horizon is not visible.
    """
    small = cv2.resize(gray, (0, 0), fx=.5, fy=.5) if max(gray.shape) > 1000 else gray
    edges = cv2.Canny(small, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55, minLineLength=max(35, min(small.shape) // 7), maxLineGap=12)
    if lines is None:
        return 0.0, 0.0
    deviations = []
    # OpenCV versions return either (N, 1, 4) or (N, 4).  Flattening the
    # leading dimensions makes the detector portable across both formats.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Difference from the nearest horizontal or vertical axis, in [-45, 45].
        deviation = ((angle + 45) % 90) - 45
        deviations.append((deviation, length))
    if len(deviations) < 4:
        return 0.0, 0.0
    values = np.array([item[0] for item in deviations])
    weights = np.array([item[1] for item in deviations])
    median = float(np.median(values))
    inliers = np.abs(values - median) < 5
    if inliers.sum() < 4:
        return 0.0, 0.0
    angle = float(np.average(values[inliers], weights=weights[inliers]))
    return abs(angle), float(inliers.mean())
