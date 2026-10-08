from __future__ import annotations

from pathlib import Path
from typing import Iterator
import cv2
import numpy as np
from PIL import Image, ImageOps


def find_images(folder: str | Path, extensions: list[str]) -> Iterator[Path]:
    allowed = {ext.lower() for ext in extensions}
    yield from (p for p in sorted(Path(folder).iterdir()) if p.is_file() and p.suffix.lower() in allowed)


def load_image(path: str | Path, max_side: int | None = None) -> np.ndarray:
    """Load an image with EXIF orientation applied as RGB; raises ValueError if invalid."""
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if max_side and max(image.size) > max_side:
                scale = max_side / max(image.size)
                image.thumbnail((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
            # PIL-backed arrays may be read-only; Torch/Transformers requires a
            # writable NumPy buffer when it converts the image to a tensor.
            return np.array(image, copy=True)
    except Exception as exc:
        raise ValueError(f"Cannot read image: {exc}") from exc


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
