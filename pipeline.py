"""Load the GPC dataset, check its split and build regression models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

COMPOSITION_FEATURES = [
    "naoh_content_kg_m3", "sodium_silicate_kg_m3", "ggbfs_kg_m3",
    "fly_ash_kg_m3", "extra_water_kg_m3", "fine_aggregate_kg_m3",
    "coarse_aggregate_kg_m3",
]
FEATURES = COMPOSITION_FEATURES + ["naoh_molarity_mol_l", "curing_temperature_c", "age_days"]
OMITTED_FEATURES = ["ss_sh_ratio", "alkaline_binder_ratio", "superplasticizer_kg_m3"]
TARGET = "strength_mpa"
SEED = 42
SOURCE = "Aravind-Unni/GPC-Dataset: local 820-row snapshot"


def mixture_keys(frame: pd.DataFrame) -> pd.Series:
    # Keep a base composition together across ages, molarities and curing temperatures.
    return frame[COMPOSITION_FEATURES].astype(float).round(8).apply(
        lambda row: hashlib.sha256(json.dumps(row.tolist(), separators=(",", ":")).encode()).hexdigest()[:16],
        axis=1,
    )


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    raw = pd.read_csv(path)
    required = ["record_id", *FEATURES, *OMITTED_FEATURES, TARGET]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    if raw.record_id.duplicated().any() or not np.array_equal(raw.record_id.to_numpy(), np.arange(len(raw))):
        raise ValueError("Record IDs must match the source row order exactly once")
    values = raw[[*FEATURES, *OMITTED_FEATURES, TARGET]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy()).all(axis=1)
    valid = finite & (values[FEATURES] >= 0).all(axis=1) & (values[TARGET] > 0) & (values.age_days > 0)
    if not valid.all():
        raise ValueError(f"Invalid dataset rows: {raw.loc[~valid, 'record_id'].tolist()}")
    data = values.copy()
    data["record_id"] = raw.record_id.astype(int)
    data["excel_row"] = data.record_id + 2
    data["sample_id"] = data.record_id.map(lambda i: f"GPC{i+1:04d}")
    data["source"] = SOURCE
    data["source_id"] = "GPC820"
    data["mixture_group"] = mixture_keys(data)
    data["group"] = data.mixture_group
    audit = {
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows_after_header": len(raw), "records_retained": len(data),
        "features": FEATURES, "omitted_predictors": OMITTED_FEATURES, "target": TARGET,
        "source_labels": 1, "row_level_study_mapping_available": False,
        "mixture_groups": int(data.group.nunique()), "grouping_fields": COMPOSITION_FEATURES,
        "grouping_precision_decimals": 8,
        "recorded_zero_molarity_rows": int(data.naoh_molarity_mol_l.eq(0).sum()),
        "identical_selected_input_output_repeats_retained": int(data.duplicated([*FEATURES, TARGET]).sum()),
        "identical_selected_input_rows": int(data.duplicated(FEATURES).sum()),
        "excluded_rows": [],
        "cleaning_policy": "All 820 source rows retained; numeric values and source codes preserved.",
    }
    sources = pd.DataFrame([{"source_id": "GPC820", "source": SOURCE, "records": len(data),
                             "url": "https://github.com/Aravind-Unni/GPC-Dataset",
                             "row_level_study_mapping": "not supplied"}])
    return data, audit, sources


def deterministic_holdout(data: pd.DataFrame):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    return next(splitter.split(data, groups=data.group))


def load_holdout(data: pd.DataFrame, audit: dict, data_dir: Path):
    """Check the pinned snapshot, row assignments and deterministic group split."""
    manifest = json.loads((data_dir / "source_manifest.json").read_text(encoding="utf-8"))
    dataset_name = manifest["dataset"]
    if audit["raw_sha256"] != manifest["files"][dataset_name]:
        raise ValueError("Dataset hash differs from the pinned course snapshot")
    for name in ("holdout_assignment.csv", "holdout_metadata.csv"):
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Changed supplied split file: {name}")
    assignment = pd.read_csv(data_dir / "holdout_assignment.csv")
    if assignment.sample_id.duplicated().any() or set(assignment.sample_id) != set(data.record_id):
        raise ValueError("Holdout IDs must match the dataset exactly once")
    assignment = assignment.set_index("sample_id").loc[data.record_id]
    if not np.array_equal(assignment.group.to_numpy(), data.group.to_numpy()):
        raise ValueError("Stored mixture groups do not match the dataset")
    flags = assignment.is_holdout.astype(str).str.lower().map({"true": True, "false": False})
    if flags.isna().any():
        raise ValueError("Invalid holdout flags")
    held = flags.to_numpy(dtype=bool)
    train_idx, test_idx = np.flatnonzero(~held), np.flatnonzero(held)
    expected_train, expected_test = deterministic_holdout(data)
    if set(test_idx) != set(expected_test) or set(train_idx) != set(expected_train):
        raise ValueError("Stored holdout does not match the fixed seed-42 group split")
    meta = pd.read_csv(data_dir / "holdout_metadata.csv").iloc[0].to_dict()
    expected = {"n_samples_total": len(data), "n_samples_holdout": len(test_idx),
                "n_mixes_total": int(data.group.nunique()),
                "n_mixes_holdout": int(data.iloc[test_idx].group.nunique()), "random_state": SEED}
    for key, value in expected.items():
        if int(meta[key]) != value:
            raise ValueError(f"Split metadata mismatch: {key}")
    if set(data.iloc[train_idx].group) & set(data.iloc[test_idx].group):
        raise ValueError("The supplied split leaks base compositions")
    return train_idx, test_idx, meta


def make_models() -> dict:
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=2),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            random_state=SEED, n_jobs=2, verbosity=0, tree_method="hist"),
    }


def check_prediction_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if not len(frame):
        raise ValueError("Prediction input contains no rows")
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError("Predictors must be finite non-negative numeric values")
    if (values.age_days <= 0).any():
        raise ValueError("Age must be positive; it is specimen age, not zero-day exposure")
    return values
