"""Local CLIP image/text encoder. Model weights are cached by Hugging Face."""
from __future__ import annotations
import numpy as np


class ClipModel:
    def __init__(self, model_name: str, device: str = "auto"):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Install transformers and torch to use CLIP.") from exc
        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            vectors = self.model.get_text_features(**inputs)
        return self._normalize(self._to_numpy(vectors))

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            vectors = self.model.get_image_features(**inputs)
        return self._normalize(self._to_numpy(vectors))[0]

    @staticmethod
    def _to_numpy(output) -> np.ndarray:
        """Accept both tensor (Transformers 4.x) and ModelOutput (5.x) results."""
        tensor = output
        if not hasattr(tensor, "detach"):
            # Transformers 5 returns BaseModelOutputWithPooling.  Its projected
            # CLIP feature vector is deliberately stored in pooler_output.
            tensor = getattr(output, "pooler_output", None)
        if tensor is None or not hasattr(tensor, "detach"):
            raise TypeError(f"CLIP returned an unsupported feature output: {type(output).__name__}")
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
