"""Reproduce the State Horizon decision threshold from chronological OOF scores."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "docs" / "state_horizon_oof_predictions.csv"
OUTPUT_PATH = ROOT / "docs" / "state_horizon_threshold_metrics.csv"


def main() -> None:
    oof = pd.read_csv(OOF_PATH)
    y_true = oof["target"].astype(int)
    probability = oof["state_horizon_probability"]
    # Cutoffs are observed OOF probabilities. On a tie, keep the lower cutoff.
    threshold = max(
        sorted(probability.unique()),
        key=lambda t: (f1_score(y_true, probability >= t), -t),
    )
    predicted = probability >= threshold
    metrics = pd.DataFrame([{
        "model": "State Horizon",
        "metric_scope": "pooled chronological OOF",
        "n_rows": len(oof),
        "threshold": threshold,
        "f1": f1_score(y_true, predicted),
        "precision": precision_score(y_true, predicted),
        "recall": recall_score(y_true, predicted),
    }])
    metrics.to_csv(OUTPUT_PATH, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
