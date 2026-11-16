"""Reproduce from copied inputs/code in an empty directory and compare predictions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)
    checked = []
    with tempfile.TemporaryDirectory(prefix="cv7001_reproduction_") as temporary:
        stage = Path(temporary)
        for name in ("pipeline.py", "train.py", "predict.py", "analyze_results.py", "plot_results.py"):
            shutil.copy2(ROOT/name, stage/name)
        for name in ("data", "examples"):
            shutil.copytree(ROOT/name, stage/name)
        runs = []
        for name in ("train.py", "predict.py", "analyze_results.py"):
            run = subprocess.run([sys.executable, name], cwd=stage, check=True,
                                 capture_output=True, text=True, encoding="utf-8")
            runs.append({"command": f"python {name}", "stdout": run.stdout, "stderr": run.stderr})
        for name in ("metrics.csv", "test_predictions.csv", "train_predictions.csv", "example_predictions.csv",
                     "strength_bands.csv", "error_coverage.csv", "development_permutation_importance.csv", "permutation_folds.csv"):
            expected, actual = pd.read_csv(ROOT/"results"/name), pd.read_csv(stage/"results"/name)
            pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-10, atol=1e-10)
            checked.append(name)
        pd.testing.assert_frame_equal(pd.read_csv(ROOT/"results/split_assignments.csv"),
                                      pd.read_csv(stage/"results/split_assignments.csv"))
        audit = json.loads((stage/"results/data_audit.json").read_text(encoding="utf-8"))
    report = {"status": "passed", "method": "Fresh-directory training and inference from copied code, data and examples",
              "commands": runs, "compared_files": checked + ["split_assignments.csv"],
              "numeric_tolerance": {"rtol": 1e-10, "atol": 1e-10}, "dataset_sha256": audit["raw_sha256"],
              "environment": audit["environment"], "code_sha256": audit["code_sha256"],
              "unit_checks": 6, "scope": "Same machine and pinned Python environment"}
    (ROOT/"results/reproducibility_check.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Reproduction passed: fresh training and inference match the saved metrics, split and predictions.")


if __name__ == "__main__":
    main()
