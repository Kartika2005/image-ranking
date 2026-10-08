from __future__ import annotations
import numpy as np


def select_diverse(items: list[dict], threshold: float) -> list[dict]:
    """Greedily retain highest scoring item, then reject CLIP-near duplicates."""
    selected: list[dict] = []
    for item in sorted(items, key=lambda row: row["final_score"], reverse=True):
        similarities = [float(np.dot(item["embedding"], kept["embedding"])) for kept in selected]
        nearest = max(similarities, default=0.)
        item["nearest_similarity"] = nearest
        if nearest < threshold:
            item["diversity_score"] = 100. * (1 - nearest)
            selected.append(item)
    return selected


def balance_categories(items: list[dict], top_k: int, max_per_category: int) -> list[dict]:
    """Prefer event-scene breadth, then fill spare places by score.

    This is deliberately applied only after near-duplicate removal. It prevents a
    long set of technically good audience images from crowding out detected speaker,
    stage, award-distribution, or group-photo moments.
    """
    selected, deferred, counts = [], [], {}
    for item in items:
        category = item["predicted_category"]
        if counts.get(category, 0) < max_per_category and len(selected) < top_k:
            selected.append(item)
            counts[category] = counts.get(category, 0) + 1
        else:
            deferred.append(item)
    return (selected + deferred)[:top_k]
