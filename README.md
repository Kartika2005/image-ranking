# Seminar Image Ranker

An explainable, offline-after-download computer-vision pipeline for choosing representative seminar/event photographs. It uses pretrained CLIP embeddings for zero-shot scene similarity, YOLO for people/object signals, OpenCV quality metrics, and greedy CLIP-embedding duplicate removal. It does **not** train a classifier, use an LLM, send images to an API, or claim to be trained to identify the best photograph.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Enter an image folder (or upload files), choose Top-K, and analyze. The first run downloads open pretrained CLIP and YOLO weights into their normal local caches. Subsequent runs can be offline.

## Output

`output/ranked/` contains copied selections prefixed with their rank. `output/results.csv` includes scores, features, category, and rule-generated explanation. Corrupt/unreadable files are skipped and reported.

## Scoring

Every component is 0–100. `final_score` is:

`0.35 semantic relevance + 0.30 image quality + 0.20 event coverage + 0.15 composition`

Semantic relevance blends zero-shot CLIP confidence with the configured category priority; it is not a standalone winner. The category priorities, weights, Top-K, models, and duplicate cosine threshold all live in `config/config.yaml`. Duplicate filtering greedily keeps the higher-scored candidate whenever its normalized CLIP image embedding has cosine similarity at or above the configured threshold.

After duplicate removal, `max_per_category_in_results` ensures one scene type cannot occupy the whole Top-K. The highest scoring candidate still comes first; deferred same-category candidates fill open places only when the other detected scene categories have had an opportunity to appear.

Before ranking, configurable eligibility gates exclude clearly blurry frames, reliably detected severe roll/tilt, low technical-quality frames, and weakly composed frames. The tilt check uses room-line geometry, with its threshold and minimum confidence configurable. Each semantic category has multiple focused CLIP prompts: speakers, speaker-plus-audience, and certificate/prize distributions are explicitly represented; the latter maps to `EVENT_ACTIVITY`.

YOLO's standard COCO model reliably provides `person` and may provide `laptop`. It has no native `microphone` or `podium` class, so those remain false unless a compatible custom model is configured—no detections are fabricated.

## Future supervised version

When human ranking labels exist, preserve the feature extraction modules and replace `ranking/scoring.py` with a learned ranker such as XGBoost or Random Forest.
