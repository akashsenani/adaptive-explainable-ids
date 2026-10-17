"""Create small feature-only explanation backgrounds for deployment.

Run from the original X-AI PROJECT root (the folder containing 0_Datasets):
    python adaptive-ids-deploy/prepare_backgrounds.py

Only up to 100 feature rows per dataset are copied into the deployment folder (sampled from the first 5,000 training rows to avoid loading full large files).
No labels or raw dataset files are copied.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "adaptive-ids-deploy"
REGISTRY = json.loads((DEPLOY / "model_registry.json").read_text(encoding="utf-8"))

DATASETS = {
    "NSL-KDD": ROOT / "0_Datasets" / "NSL-KDD dataset" / "nsl-kdd" / "train.csv",
    "Farm-Flow": ROOT / "0_Datasets" / "Farm-Flow" / "Datasets" / "train.csv",
    "IoMT-AI-IDS": ROOT / "0_Datasets" / "ML-Based IDS IoMT" / "Dataset & Captures" / "Datasets" / "IP-Based" / "Flows" / "train.csv",
}

out = DEPLOY / "backgrounds"
out.mkdir(parents=True, exist_ok=True)
for dataset, source in DATASETS.items():
    if not source.exists():
        raise FileNotFoundError(f"Training CSV not found for {dataset}: {source}")
    features = REGISTRY[dataset]["features"]
    # Read only the feature columns needed for explanation. Use a reproducible small sample.
    frame = pd.read_csv(source, usecols=lambda c: c in set(features), nrows=5000)
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(f"{dataset} CSV is missing expected model features: {missing}")
    frame = frame[features].sample(n=min(100, len(frame)), random_state=42)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target = out / f"{dataset}_background.csv"
    frame.to_csv(target, index=False)
    print(f"Created {target} with {len(frame)} rows and {len(features)} features")
print("Done. Only feature columns from small samples were copied; no labels were included.")
