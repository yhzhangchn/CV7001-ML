"""Extract the existing 820-row worksheet and pin one grouped holdout (no training)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import FEATURES, OMITTED_FEATURES, SEED, deterministic_holdout, load_dataset

ROOT = Path(__file__).resolve().parent
COLUMNS = ["fly_ash_kg_m3", "ggbfs_kg_m3", "coarse_aggregate_kg_m3", "fine_aggregate_kg_m3",
           "naoh_content_kg_m3", "naoh_molarity_mol_l", "sodium_silicate_kg_m3", "ss_sh_ratio",
           "alkaline_binder_ratio", "age_days", "curing_temperature_c", "extra_water_kg_m3",
           "superplasticizer_kg_m3", "strength_mpa"]


def prepare(workbook: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(workbook, sheet_name="Raw_GitHub_820").iloc[:, :14].dropna(how="all")
    if len(raw) != 820 or raw.shape[1] != 14:
        raise ValueError("Expected the 820-row, 13-predictor public dataset worksheet")
    if "Fly Ash" not in str(raw.columns[0]) or "Compressive Strength" not in str(raw.columns[-1]):
        raise ValueError("Unexpected worksheet schema")
    original_headers = [str(x).strip() for x in raw.columns]
    raw.columns = COLUMNS
    raw = raw.apply(pd.to_numeric, errors="raise")
    raw.insert(0, "record_id", np.arange(len(raw)))
    dataset = output / "gpc_database_820.csv"
    raw.to_csv(dataset, index=False, float_format="%.15g")
    data, audit, _ = load_dataset(dataset)
    train_idx, test_idx = deterministic_holdout(data)
    assignment = pd.DataFrame({"sample_id": data.record_id, "group": data.group, "is_holdout": False})
    assignment.loc[test_idx, "is_holdout"] = True
    assignment.to_csv(output / "holdout_assignment.csv", index=False)
    meta = {"split_strategy": "GroupShuffleSplit_20pct_base_compositions", "random_state": SEED,
            "holdout_fraction": 0.2, "n_mixes_total": int(data.group.nunique()),
            "n_mixes_holdout": int(data.iloc[test_idx].group.nunique()),
            "n_samples_total": len(data), "n_samples_holdout": len(test_idx)}
    pd.DataFrame([meta]).to_csv(output / "holdout_metadata.csv", index=False)
    manifest = {"dataset": dataset.name,
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in (dataset.name, "holdout_assignment.csv", "holdout_metadata.csv")},
        "source": {"local_workbook": workbook.name, "local_workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
                   "worksheet": "Raw_GitHub_820", "repository": "https://github.com/Aravind-Unni/GPC-Dataset",
                   "provenance": "Previously downloaded local copy of the public GPC dataset",
                   "original_headers": original_headers, "normalised_headers": COLUMNS},
        "feature_policy": {"selected": FEATURES, "omitted": OMITTED_FEATURES,
                           "decision": "Ten inputs selected before training: material amounts, molarity, temperature and age."}}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Prepared {len(data)} records, {audit['mixture_groups']} base compositions; dev/test {len(train_idx)}/{len(test_idx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    prepare(args.workbook.resolve(), args.output_dir.resolve())
