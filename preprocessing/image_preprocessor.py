from __future__ import annotations
import numpy as np
from utils.image_utils import load_image


class ImagePreprocessor:
    def __init__(self, max_side: int = 1280):
        self.max_side = max_side

    def load(self, path: str) -> np.ndarray:
        return load_image(path, self.max_side)
