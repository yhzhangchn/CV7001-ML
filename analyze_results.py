"""Summarise strength-range errors and development-fold feature reliance."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error

from pipeline import FEATURES, TARGET, SEED, load_dataset, load_holdout, make_models

ROOT = Path(__file__).resolve().parent


def run():
    output = ROOT / "results"
    data, audit, _ = load_dataset(ROOT / "data/gpc_database_820.csv")
    train_idx, test_idx, _ = load_holdout(data, audit, ROOT / "data")
    assignments = pd.read_csv(output / "split_assignments.csv").set_index("record_id")
    metrics = pd.read_csv(output / "metrics.csv")
    selected = metrics.loc[metrics.selected_by_cv.astype(bool)].iloc[0]
    name = selected["model"]
    dev = data.iloc[train_idx].copy()
    dev["validation_fold"] = dev.record_id.map(assignments.validation_fold)
    reference_folds = pd.read_csv(output / "cv_folds.csv")
    permutation_rows = []
    for fold in range(1, 6):
        fit = dev.loc[dev.validation_fold.ne(fold)]
        val = dev.loc[dev.validation_fold.eq(fold)]
        if set(fit.group) & set(val.group):
            raise AssertionError("Overlapping composition groups")
        model = clone(make_models()[name]).fit(fit[FEATURES], fit[TARGET])
        baseline = root_mean_squared_error(val[TARGET], model.predict(val[FEATURES]))
        reference = reference_folds.loc[(reference_folds.model == name) & (reference_folds.fold == fold), "RMSE"].iloc[0]
        np.testing.assert_allclose(baseline, reference, atol=1e-6, rtol=0)
        importance = permutation_importance(model, val[FEATURES], val[TARGET],
            scoring="neg_root_mean_squared_error", n_repeats=10, random_state=SEED + fold, n_jobs=1)
        for column, values in zip(FEATURES, importance.importances):
            for repeat, value in enumerate(values, 1):
                permutation_rows.append({"fold": fold, "feature": column, "repeat": repeat,
                    "baseline_RMSE": float(baseline), "delta_RMSE": float(value)})
    raw_permutation = pd.DataFrame(permutation_rows)
    fold_means = raw_permutation.groupby(["fold", "feature"], as_index=False).delta_RMSE.mean()
    importance = fold_means.groupby("feature").delta_RMSE.agg(["mean", "std"]).reset_index()
    importance.columns = ["feature", "delta_RMSE_mean", "fold_mean_SD"]
    importance = importance.sort_values("delta_RMSE_mean", ascending=False).reset_index(drop=True)

    predictions = pd.read_csv(output / "test_predictions.csv")
    expected = data.iloc[test_idx].set_index("record_id")
    actual_ids = predictions.sample_id.str.removeprefix("GPC").astype(int) - 1
    if set(actual_ids) != set(expected.index):
        raise AssertionError("Test predictions do not match the fixed split")
    np.testing.assert_allclose(predictions[TARGET], actual_ids.map(expected[TARGET]), atol=1e-12)
    truth = predictions[TARGET]
    error = predictions[name] - truth
    bands = []
    masks = [truth.lt(20), truth.ge(20) & truth.lt(60), truth.ge(60)]
    train_truth = dev[TARGET]
    train_masks = [train_truth.lt(20), train_truth.ge(20) & train_truth.lt(60), train_truth.ge(60)]
    for band, mask, training_mask in zip(["<20", "20-<60", ">=60"], masks, train_masks):
        e = error.loc[mask]
        bands.append({"strength_band_MPa": band, "n_test": int(mask.sum()),
            "n_development": int(training_mask.sum()), "MAE": float(e.abs().mean()),
            "RMSE": float(np.sqrt(np.mean(e**2))), "mean_error": float(e.mean())})
    band_frame = pd.DataFrame(bands)
    coverage = pd.DataFrame([{"absolute_error_limit_MPa": limit, "n_within": int(error.abs().le(limit).sum()),
        "n_total": len(error), "fraction_within": float(error.abs().le(limit).mean())} for limit in (5, 10)])

    raw_permutation.to_csv(output / "permutation_folds.csv", index=False)
    importance.to_csv(output / "development_permutation_importance.csv", index=False)
    band_frame.to_csv(output / "strength_bands.csv", index=False)
    coverage.to_csv(output / "error_coverage.csv", index=False)
    by_name = metrics.set_index("model")
    summary = {"selected_model": name, "permutation_scope": "Five development validation folds; ten shuffles per feature in each fold",
        "importance_unit": "Increase in RMSE (MPa), averaged equally over folds",
        "permutation_model_settings": "Same fixed settings and folds as the original training run",
        "ridge_RMSE_reduction_pct": float(100 * (1 - selected.test_RMSE / by_name.loc["Ridge", "test_RMSE"])),
        "RF_RMSE_reduction_pct": float(100 * (1 - selected.test_RMSE / by_name.loc["Random Forest", "test_RMSE"])),
        "strength_bands": bands, "error_coverage": coverage.to_dict("records"),
        "development_importance": importance.to_dict("records"),
        "max_absolute_test_error": float(error.abs().max()),
        "input_file_sha256": {filename: hashlib.sha256((output / filename).read_bytes()).hexdigest()
            for filename in ("test_predictions.csv", "metrics.csv", "split_assignments.csv", "cv_folds.csv")},
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(importance.to_string(index=False))
    print(band_frame.to_string(index=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    run()
