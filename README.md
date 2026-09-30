# IndabaX South Sudan 2026 — Food Insecurity Forecasting

Author: David Ngang Mabior

I developed this solution for the IndabaX South Sudan 2026 Food Insecurity Forecasting Challenge. The task is to forecast a binary, county-period risk of Crisis-or-worse food insecurity (IPC Phase 3+); the competition metric is ROC-AUC.

I use only the organizer-provided `Train.csv`, `Test.csv`, and `SampleSubmission.csv` files—no external datasets, AutoML, hidden test labels, or leaderboard feedback. My two selected final approaches are **State Horizon** and **Interval Trend**. Both are deterministic HistGradientBoosting pipelines with `random_state=42` and single-thread environment settings.

Historical county and state features are constructed under a strict temporal rule: for a prediction period T, they use only periods earlier than T. I evaluated the approaches with chronological forward validation, not random splitting. Predictions are keyed to ID and explicitly reindexed to the sample submission IDs before writing.

## Data setup

Obtain the organizer files and place them exactly as documented in [data/README.md](data/README.md). The data are deliberately excluded from this repository.

Use Python 3.13 (the reviewed environment used Python 3.13.15), then install:

```powershell
python -m pip install -r requirements.txt
```

## Reproduce the final outputs

From the repository root, run:

```powershell
python src\research\final_champion_runner.py --name state_horizon --output outputs\state_horizon.csv
python src\research\final_champion_runner.py --name interval_trend --output outputs\interval_trend.csv
python src\research\calculate_state_horizon_threshold.py
```

The first two commands write `outputs/state_horizon.csv` and `outputs/interval_trend.csv`. The final command reproduces the State Horizon decision-threshold metrics from the included chronological OOF predictions.

The finalist submissions received a public leaderboard ROC-AUC of 1.000. That result is a ranking metric, not 100% accuracy and not evidence of perfect real-world forecasting. This work is intended as a reproducible forecasting and prioritization tool, not a replacement for IPC analysis.

See [the technical write-up](docs/TECHNICAL_WRITEUP.md) for features, validation, threshold selection, and humanitarian interpretation.
