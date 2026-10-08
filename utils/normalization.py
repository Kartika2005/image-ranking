"""Small, explicit normalization helpers used by the ranking pipeline."""
from __future__ import annotations

import math


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def linear(value: float, low: float, high: float) -> float:
    """Map a value to 0..100, clipping values outside its expected range."""
    if high <= low:
        return 0.0
    return clamp((value - low) * 100.0 / (high - low))


def bell(value: float, target: float, tolerance: float) -> float:
    """A transparent 0..100 score, highest at target and falling smoothly."""
    if tolerance <= 0:
        return 0.0
    return clamp(100.0 * math.exp(-0.5 * ((value - target) / tolerance) ** 2))
