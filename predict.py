"""Predict strength for CSV rows using the saved CV-selected course model."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pipeline import check_prediction_input

ROOT = Path(__file__).resolve().parent


def predict(frame, bundle):
    values = check_prediction_input(frame)
    result = frame.copy()
    result["predicted_strength_mpa"] = np.asarray(bundle["model"].predict(values), dtype=np.float64)
    result["outside_training_feature_range"] = (
        values.lt(pd.Series(bundle["training_min"])) |
        values.gt(pd.Series(bundle["training_max"]))
    ).any(axis=1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples" / "new_mixtures.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "results" / "models" / "selected.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "example_predictions.csv")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    prediction = predict(pd.read_csv(args.input), bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(prediction[["predicted_strength_mpa", "outside_training_feature_range"]].to_string(index=False))
