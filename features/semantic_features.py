from __future__ import annotations
import numpy as np


def extract_semantics(image_embedding: np.ndarray, text_embeddings: np.ndarray, categories: dict[str, list[str]]) -> dict:
    """Use the strongest specific prompt for each event scene category."""
    similarities = image_embedding @ text_embeddings.T
    scores, offset = {}, 0
    for category, prompts in categories.items():
        count = len(prompts)
        scores[category] = float(similarities[offset:offset + count].max())
        offset += count
    predicted = max(scores, key=scores.get)
    confidence = scores[predicted]
    return {"category_scores": scores, "predicted_category": predicted, "semantic_confidence": confidence,
            "semantic_score": max(0., min(100., (confidence + 0.05) / .45 * 100))}
