"""Orchestrates the local pretrained-CV + explainable feature-engineering pipeline."""
from __future__ import annotations
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import yaml

from features.composition_features import extract_composition
from features.object_features import event_coverage_score, extract_objects
from features.quality_features import extract_quality
from features.semantic_features import extract_semantics
from models.clip_model import ClipModel
from models.yolo_model import YoloModel
from preprocessing.image_preprocessor import ImagePreprocessor
from ranking.diversity import balance_categories, select_diverse
from ranking.scoring import calculate_final_score
from utils.image_utils import find_images


@dataclass
class RankResult:
    selected: list[dict]
    processed_count: int
    errors: list[str]
    rejected: list[str]
    output_dir: Path


class SeminarImageRanker:
    def __init__(self, config_path: str | Path = "config/config.yaml"):
        with open(config_path, encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        settings = self.config["processing"]
        self.preprocessor = ImagePreprocessor(settings["max_analysis_side"])
        self.clip = ClipModel(self.config["models"]["clip_name"], self.config["models"]["device"])
        self.yolo = YoloModel(self.config["models"]["yolo_name"], settings["yolo_confidence"])
        self.categories = self.config["categories"]
        prompts = [prompt for category_prompts in self.categories.values() for prompt in category_prompts]
        self.text_embeddings = self.clip.encode_text(prompts)

    def rank(self, input_folder: str | Path, top_k: int | None = None, output_dir: str | Path = "output",
             progress: Callable[[int, int, str], None] | None = None) -> RankResult:
        paths = list(find_images(input_folder, self.config["processing"]["image_extensions"]))
        if not paths:
            raise ValueError("No supported images found. Use JPG, JPEG, PNG, HEIC, or HEIF files.")
        rows, errors, rejected = [], [], []
        for index, path in enumerate(paths, 1):
            try:
                image = self.preprocessor.load(str(path))
                embedding = self.clip.encode_image(image)
                semantic = extract_semantics(embedding, self.text_embeddings, self.categories)
                objects = extract_objects(self.yolo.detect(image), image.shape)
                quality = extract_quality(image)
                composition = extract_composition(objects, image.shape)
                row = {"source_path": str(path), "filename": path.name, "embedding": embedding, **semantic, **objects, **quality, **composition}
                row["event_coverage_score"] = event_coverage_score(objects)
                row = calculate_final_score(row, self.config)
                eligible, reason = self._is_eligible(row)
                if eligible:
                    rows.append(row)
                else:
                    rejected.append(f"{path.name}: {reason}")
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
            if progress:
                progress(index, len(paths), path.name)
        diverse = select_diverse(rows, self.config["processing"]["duplicate_similarity_threshold"])
        requested_k = top_k or self.config["processing"]["top_k"]
        selected = balance_categories(diverse, requested_k, self.config["processing"].get("max_per_category_in_results", requested_k))
        for rank, row in enumerate(selected, 1):
            row["rank"] = rank
            row["explanation"] = make_explanation(row)
        destination = Path(output_dir)
        self._write_output(selected, destination)
        return RankResult(selected, len(rows), errors, rejected, destination)

    def _is_eligible(self, row: dict) -> tuple[bool, str]:
        """Hard technical gates ensure selections have a usable, level frame."""
        limits = self.config["processing"].get("eligibility", {})
        if row["sharpness"] < limits.get("min_sharpness", 0):
            return False, f"excluded as blurry (sharpness {row['sharpness']:.1f})"
        # Tilt is only trusted if enough structural lines agreed on the estimate.
        if (row["tilt_confidence"] >= limits.get("min_tilt_confidence", .15)
                and row["tilt_degrees"] > limits.get("max_tilt_degrees", 90)):
            return False, f"excluded as tilted ({row['tilt_degrees']:.1f}°)"
        if row["quality_score"] < limits.get("min_quality_score", 0):
            return False, f"excluded for weak technical quality ({row['quality_score']:.1f})"
        if row["composition_score"] < limits.get("min_composition_score", 0):
            return False, f"excluded for improper framing ({row['composition_score']:.1f})"
        return True, ""

    @staticmethod
    def _write_output(rows: list[dict], output_dir: Path) -> None:
        ranked = output_dir / "ranked"
        ranked.mkdir(parents=True, exist_ok=True)
        for row in rows:
            shutil.copy2(row["source_path"], ranked / f"{row['rank']:02d}_{row['filename']}")
        columns = ["rank", "filename", "final_score", "predicted_category", "semantic_score", "quality_score", "sharpness", "brightness", "contrast", "tilt_degrees", "person_count", "person_area_ratio", "diversity_score", "event_coverage_score", "composition_score", "explanation"]
        with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def make_explanation(row: dict) -> str:
    reasons = [f"{row['predicted_category'].replace('_', ' ').title()} is a high-priority event scene"]
    if row["sharpness_score"] >= 65:
        reasons.append("high sharpness")
    if row["brightness_score"] >= 70:
        reasons.append("good brightness")
    if row["person_count"]:
        reasons.append(f"{row['person_count']} people detected")
    if row["event_coverage_score"] >= 65:
        reasons.append("strong event coverage")
    if row.get("nearest_similarity", 0) < .5:
        reasons.append("visually distinct from higher-ranked selections")
    return "; ".join(reasons) + "."
