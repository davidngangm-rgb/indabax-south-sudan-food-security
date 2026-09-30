"""Canonical leakage-safe utilities for South Sudan competition research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESEARCH_OUTPUTS = ROOT / "research_outputs"
EXPERIMENT_DIR = RESEARCH_OUTPUTS / "experiments"
SUBMISSION_DIR = RESEARCH_OUTPUTS / "submissions"
RANDOM_STATE = 42
VALIDATION_PERIODS = ("2023-09", "2023-12", "2024-04", "2024-09", "2024-12", "2025-04")
IPC_MAP = {"Minimal": 1, "Stressed": 2, "Crisis": 3, "Emergency": 4, "Catastrophe": 5}


def load_competition_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load immutable competition files and validate the sample-ID contract."""
    # Organizer files are renamed only to these stable, documented paths.
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    assert train["ID"].is_unique and test["ID"].is_unique and sample["ID"].is_unique
    assert sample["ID"].equals(test["ID"])
    assert list(sample.columns) == ["ID", "food_insecurity_risk"]
    return train, test, sample


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["period"] = pd.to_datetime(dict(year=out.start_year, month=out.start_month, day=1))
    out["prior_ipc_numeric"] = out["prior_period_ipc_phase"].map(IPC_MAP).astype(float)
    out["prior_ipc_crisis_plus"] = (out["prior_ipc_numeric"] >= 3).astype(int)
    out["prior_ipc_emergency_plus"] = (out["prior_ipc_numeric"] >= 4).astype(int)
    population = out["population"].replace(0, np.nan)
    production = out["prior_year_cereal_production_tonnes"].replace(0, np.nan)
    out["production_per_capita"] = out["prior_year_cereal_production_tonnes"] / population
    out["gap_per_capita"] = out["prior_year_cereal_gap_tonnes"] / population
    out["gap_to_production"] = out["prior_year_cereal_gap_tonnes"] / production
    angle = 2 * np.pi * out["start_month"] / 12.0
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    out["absolute_gap"] = out["prior_year_cereal_gap_tonnes"].abs()
    out["ipc_phase_x_phase3pct"] = out["prior_ipc_numeric"] * out["prior_period_phase3plus_pct"]
    out["ipc_phase_x_gap_pc"] = out["prior_ipc_numeric"] * out["gap_per_capita"]
    out["phase3pct_x_gap_pc"] = out["prior_period_phase3plus_pct"] * out["gap_per_capita"]
    return out


HISTORY_BASE_COLUMNS = (
    "prior_ipc_numeric", "prior_period_phase3plus_pct",
    "prior_year_cereal_production_tonnes", "prior_year_cereal_gap_tonnes",
    "production_per_capita", "gap_per_capita", "gap_to_production",
)


def add_strict_period_history(df: pd.DataFrame, columns: Iterable[str] = HISTORY_BASE_COLUMNS) -> pd.DataFrame:
    """Add only county history from periods strictly less than the row period.

    Same-period duplicate rows are first collapsed to a county-period mean for
    later history. They are never ordered against, lagged from, or used as
    history for one another.
    """
    out = df.copy()
    columns = list(columns)
    period_level = (
        out.groupby(["county", "period"], as_index=False, sort=True)[columns]
        .mean()
        .sort_values(["county", "period"], kind="stable")
        .reset_index(drop=True)
    )
    group = period_level.groupby("county", sort=False)
    period_level["county_history_period_count"] = group.cumcount()
    for col in columns:
        shifted = group[col].shift(1)
        period_level[f"hist_{col}_mean"] = shifted.groupby(period_level["county"]).transform(lambda s: s.expanding().mean())
        period_level[f"{col}_lag1"] = shifted
        for lag in (2, 3, 4, 5):
            period_level[f"{col}_lag{lag}"] = group[col].shift(lag)
        period_level[f"hist_{col}_recent3"] = group[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        period_level[f"hist_{col}_recent5"] = group[col].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        period_level[f"{col}_change_1"] = period_level[f"{col}_lag1"] - period_level[f"{col}_lag2"]
        period_level[f"{col}_change_2"] = period_level[f"{col}_lag2"] - period_level[f"{col}_lag3"]
        period_level[f"{col}_acceleration"] = period_level[f"{col}_change_1"] - period_level[f"{col}_change_2"]
        period_level[f"{col}_recent_vs_history"] = period_level[f"{col}_lag1"] - period_level[f"hist_{col}_mean"]
    previous_period = group["period"].shift(1)
    period_level["months_since_previous_period"] = (
        (period_level["period"].dt.year - previous_period.dt.year) * 12
        + period_level["period"].dt.month - previous_period.dt.month
    ).replace(0, np.nan)
    # Change between the two strictly-prior observations, expressed per month.
    lag_interval = (
        (group["period"].shift(1).dt.year - group["period"].shift(2).dt.year) * 12
        + group["period"].shift(1).dt.month - group["period"].shift(2).dt.month
    ).replace(0, np.nan)
    period_level["prior_history_interval_months"] = lag_interval
    for col in columns:
        period_level[f"{col}_change_per_prior_month"] = period_level[f"{col}_change_1"] / lag_interval
    history_columns = [c for c in period_level.columns if c not in {"county", "period", *columns}]
    return out.merge(period_level[["county", "period", *history_columns]], on=["county", "period"], how="left", validate="many_to_one")


def add_strict_state_history(df: pd.DataFrame, columns: Iterable[str] = HISTORY_BASE_COLUMNS) -> pd.DataFrame:
    """State aggregates from periods strictly before T, never from period T."""
    out = df.copy()
    columns = list(columns)
    state_period = (
        out.groupby(["state", "period"], as_index=False, sort=True)[columns].mean()
        .sort_values(["state", "period"], kind="stable").reset_index(drop=True)
    )
    group = state_period.groupby("state", sort=False)
    for col in columns:
        shifted = group[col].shift(1)
        state_period[f"state_hist_{col}_mean"] = shifted.groupby(state_period["state"]).transform(lambda s: s.expanding().mean())
        state_period[f"state_{col}_lag1"] = shifted
        state_period[f"state_{col}_change_1"] = shifted - group[col].shift(2)
        state_period[f"state_{col}_recent3"] = group[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    feature_columns = [c for c in state_period if c not in {"state", "period", *columns}]
    return out.merge(state_period[["state", "period", *feature_columns]], on=["state", "period"], how="left", validate="many_to_one")


def prepare_safe_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create covariate-only features with strict prior-period county history."""
    train_work = train.copy()
    test_work = test.copy()
    train_work["_is_train"] = 1
    test_work["_is_train"] = 0
    test_work["food_insecurity_risk"] = np.nan
    combined = pd.concat([train_work, test_work], ignore_index=True, sort=False)
    combined = add_basic_features(combined)
    combined = add_strict_period_history(combined)
    return (combined.loc[combined._is_train.eq(1)].copy(), combined.loc[combined._is_train.eq(0)].copy())


def add_strict_target_history(train_features: pd.DataFrame, test_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add target history from strictly earlier county periods only.

    Test targets remain unknown. The county-period target is averaged only for
    prior *training* periods, so duplicate records at a period cannot expose
    their labels to each other and no test label can ever enter a feature.
    """
    combined = pd.concat([train_features.copy(), test_features.copy()], ignore_index=True, sort=False)
    target_period = (
        train_features.groupby(["county", "period"], as_index=False, sort=True)["food_insecurity_risk"]
        .mean()
        .rename(columns={"food_insecurity_risk": "period_target_mean"})
    )
    periods = combined[["county", "period"]].drop_duplicates().merge(target_period, on=["county", "period"], how="left")
    periods = periods.sort_values(["county", "period"], kind="stable").reset_index(drop=True)
    group = periods.groupby("county", sort=False)
    shifted = group["period_target_mean"].shift(1)
    periods["target_prior_period_mean"] = shifted
    periods["target_history_mean"] = shifted.groupby(periods["county"]).transform(lambda s: s.expanding().mean())
    periods["target_history_recent3"] = group["period_target_mean"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    periods["target_history_count"] = shifted.groupby(periods["county"]).transform(lambda s: s.expanding().count())
    keep = ["county", "period", "target_prior_period_mean", "target_history_mean", "target_history_recent3", "target_history_count"]
    combined = combined.merge(periods[keep], on=["county", "period"], how="left", validate="many_to_one")
    return combined.loc[combined._is_train.eq(1)].copy(), combined.loc[combined._is_train.eq(0)].copy()


def write_submission_by_id(predictions_by_id: pd.Series, path: Path, sample: pd.DataFrame) -> None:
    """Write a submission after explicit ID-keyed alignment; never use position."""
    if not isinstance(predictions_by_id, pd.Series):
        raise TypeError("predictions_by_id must be a pandas Series indexed by ID")
    if not predictions_by_id.index.is_unique:
        raise ValueError("prediction IDs must be unique")
    aligned = predictions_by_id.reindex(sample["ID"])
    if aligned.isna().any():
        missing = sample.loc[aligned.isna(), "ID"].tolist()
        raise ValueError(f"missing predictions for IDs: {missing[:5]}")
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("predictions must be finite and within [0, 1]")
    submission = pd.DataFrame({"ID": sample["ID"], "food_insecurity_risk": values})
    assert submission["ID"].equals(sample["ID"])
    assert submission["ID"].is_unique and submission["food_insecurity_risk"].notna().all()
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def append_registry(records: list[dict]) -> None:
    """Append non-duplicate records to the central experiment ledger."""
    path = EXPERIMENT_DIR / "experiment_registry.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    incoming = pd.DataFrame(records)
    if not existing.empty:
        incoming = incoming.loc[~incoming["experiment_id"].isin(existing["experiment_id"])]
    if not incoming.empty:
        pd.concat([existing, incoming], ignore_index=True).to_csv(path, index=False)


def json_cell(value: object) -> str:
    return json.dumps(value, sort_keys=True)
