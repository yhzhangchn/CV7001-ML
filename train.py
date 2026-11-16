"""Train three fixed models on a public GPC dataset with a grouped holdout."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from pipeline import FEATURES, TARGET, SEED, load_dataset, load_holdout, make_models
from plot_results import save_prediction_figure


def metrics(y, prediction):
    return {"R2": float(r2_score(y, prediction)),
            "MAE": float(mean_absolute_error(y, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(y, prediction)))}


def check_separation(data, train_idx, test_idx):
    for column in ("mixture_group", "group"):
        if set(data.iloc[train_idx][column]) & set(data.iloc[test_idx][column]):
            raise AssertionError(f"Leaking {column} between partitions")


def run(data_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    data, audit, sources = load_dataset(data_path)
    train_idx, test_idx, split_metadata = load_holdout(data, audit, data_path.parent)
    check_separation(data, train_idx, test_idx)
    training = data.iloc[train_idx].reset_index(drop=True)
    testing = data.iloc[test_idx].reset_index(drop=True)
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(training, groups=training.group))
    audit.update({"random_seed": SEED, "train_records": len(training), "test_records": len(testing),
                  "train_groups": int(training.group.nunique()), "test_groups": int(testing.group.nunique()),
                  "train_test_source_overlap": len(set(training.source) & set(testing.source)),
                  "train_test_mixture_overlap": 0,
                  "split_rule": "Fixed seed-42 20% base-composition holdout; five grouped CV folds within development only",
                  "supplied_split_metadata": split_metadata,
                  "holdout_record_fraction": len(testing)/len(data),
                  "holdout_mixture_fraction": testing.group.nunique()/data.group.nunique(),
                  "parameter_selection": "Fixed model settings; selected by mean development-CV RMSE"})
    assignments = data[["record_id", "excel_row", "sample_id", "source_id", "source", "mixture_group", "group"]].copy()
    assignments["partition"] = "train"
    assignments.loc[test_idx, "partition"] = "test"
    assignments["validation_fold"] = 0
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        check_separation(training, fit_idx, val_idx)
        assignments.loc[train_idx[val_idx], "validation_fold"] = fold
    models = make_models()
    cv_rows = []
    for name, estimator in models.items():
        for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
            model = clone(estimator)
            model.fit(training.iloc[fit_idx][FEATURES], training.iloc[fit_idx][TARGET])
            cv_rows.append({"model": name, "fold": fold, "fit_records": len(fit_idx),
                            "validation_records": len(val_idx),
                            **metrics(training.iloc[val_idx][TARGET], model.predict(training.iloc[val_idx][FEATURES]))})
    cv_frame = pd.DataFrame(cv_rows)
    # This decision is made before any test prediction is computed.
    selected_name = str(cv_frame.groupby("model").RMSE.mean().idxmin())
    summary = []
    predictions = testing[["excel_row", "source_id", "sample_id", "group", TARGET]].copy()
    train_predictions = training[["excel_row", "source_id", "sample_id", "group", TARGET]].copy()
    for name, estimator in models.items():
        model = clone(estimator).fit(training[FEATURES], training[TARGET])
        train_prediction = np.asarray(model.predict(training[FEATURES]), dtype=np.float64)
        test_prediction = np.asarray(model.predict(testing[FEATURES]), dtype=np.float64)
        predictions[name] = test_prediction
        train_predictions[name] = train_prediction
        subset = cv_frame.loc[cv_frame.model == name]
        row = {"model": name, "selected_by_cv": name == selected_name}
        row.update({f"train_{k}": v for k, v in metrics(training[TARGET], train_prediction).items()})
        row.update({f"cv_{k}_mean": float(subset[k].mean()) for k in ("R2", "RMSE", "MAE")})
        row.update({f"cv_{k}_std": float(subset[k].std(ddof=1)) for k in ("R2", "RMSE", "MAE")})
        row.update({f"test_{k}": v for k, v in metrics(testing[TARGET], test_prediction).items()})
        summary.append(row)
        bundle = {"model": model, "name": name, "features": FEATURES,
                  "training_min": training[FEATURES].min().to_dict(),
                  "training_max": training[FEATURES].max().to_dict(), "raw_sha256": audit["raw_sha256"]}
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(bundle, output / "models" / filename)
        if name == selected_name:
            joblib.dump(bundle, output / "models" / "selected.joblib")
    summary_frame = pd.DataFrame(summary)
    audit["selected_model"] = selected_name
    audit["environment"] = {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "joblib", "openpyxl")}}
    audit["code_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                            for name in ("pipeline.py", "train.py", "predict.py", "plot_results.py")}
    summary_frame.to_csv(output / "metrics.csv", index=False)
    cv_frame.to_csv(output / "cv_folds.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    train_predictions.to_csv(output / "train_predictions.csv", index=False)
    assignments.to_csv(output / "split_assignments.csv", index=False)
    data.to_csv(output / "cleaned_data.csv", index=False)
    sources.to_csv(output / "source_index.csv", index=False)
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    truth, pred = testing[TARGET].to_numpy(), predictions[selected_name].to_numpy()
    save_prediction_figure(truth, pred, output / "heldout_predictions", selected_name)
    tex = [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in {
        "Ntotal": len(data), "Ntrain": len(training), "Ntest": len(testing),
        "Nsources": audit["source_labels"], "Ngroups": audit["mixture_groups"],
        "NtrainGroups": audit["train_groups"], "NtestGroups": audit["test_groups"],
        "ChosenModel": selected_name,
    }.items()]
    tex += ["\\newcommand{\\ResultsRows}{%"]
    for row in summary:
        tex.append(f"{row['model']} & {row['cv_RMSE_mean']:.2f} $\\pm$ {row['cv_RMSE_std']:.2f} & {row['test_R2']:.3f} & {row['test_RMSE']:.2f} & {row['test_MAE']:.2f} \\\\")
    tex += ["}"]
    (output / "metrics.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    print(summary_frame[["model", "cv_RMSE_mean", "cv_RMSE_std", "test_R2", "test_RMSE", "test_MAE"]].to_string(index=False))
    print(f"Records: {len(data)}; dev/test: {len(training)}/{len(testing)}; mixtures: {audit['mixture_groups']}; selected: {selected_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "gpc_database_820.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.data.resolve(), args.output_dir.resolve())
