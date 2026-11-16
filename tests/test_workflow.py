"""Checks for the data/split contract, independently recomputed scores and inference."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pipeline import FEATURES, OMITTED_FEATURES, TARGET, load_dataset, load_holdout, make_models, mixture_keys
from predict import predict

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.audit, _ = load_dataset(ROOT / "data/gpc_database_820.csv")
        cls.fit_idx, cls.test_idx, cls.meta = load_holdout(cls.data, cls.audit, ROOT / "data")

    def test_final_snapshot_and_fixed_holdout(self):
        self.assertEqual(len(self.data), 820)
        self.assertEqual((len(self.fit_idx), len(self.test_idx)), (670, 150))
        self.assertEqual(self.data.group.nunique(), 455)
        self.assertEqual(self.data.iloc[self.test_idx].group.nunique(), 91)
        self.assertFalse(set(self.data.iloc[self.fit_idx].group) & set(self.data.iloc[self.test_idx].group))
        self.assertEqual(len(FEATURES), 10)
        self.assertFalse(set(FEATURES) & set(OMITTED_FEATURES))
        self.assertEqual(self.audit["recorded_zero_molarity_rows"], 26)

    def test_base_composition_groups_block_age_and_processing_variants(self):
        original = self.data.iloc[[0]].copy()
        variants = pd.concat([original] * 4, ignore_index=True)
        variants.loc[1, "age_days"] += 7
        variants.loc[2, "naoh_molarity_mol_l"] += 2
        variants.loc[3, "curing_temperature_c"] += 20
        self.assertEqual(mixture_keys(variants).nunique(), 1)
        variants.loc[3, "fly_ash_kg_m3"] += 10
        self.assertEqual(mixture_keys(variants).nunique(), 2)

    def test_cv_has_no_holdout_or_mixture_overlap_and_scaler_is_local(self):
        dev = self.data.iloc[self.fit_idx].reset_index(drop=True)
        holdout_rows = set(self.data.iloc[self.test_idx].excel_row)
        cv = GroupKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in cv.split(dev, groups=dev.group):
            fit, val = dev.iloc[train_idx], dev.iloc[val_idx]
            self.assertFalse(set(fit.group) & set(val.group))
            self.assertFalse(set(fit.excel_row) & holdout_rows)
            self.assertFalse(set(val.excel_row) & holdout_rows)
            model = clone(make_models()["Ridge"]).fit(fit[FEATURES], fit[TARGET])
            np.testing.assert_allclose(model[0].mean_, fit[FEATURES].mean().to_numpy(), rtol=1e-12)

    def test_changed_assignment_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="cv7001_split_test_") as folder:
            folder = Path(folder)
            for name in ("source_manifest.json", "holdout_assignment.csv", "holdout_metadata.csv"):
                (folder/name).write_bytes((ROOT/"data"/name).read_bytes())
            a = pd.read_csv(folder/"holdout_assignment.csv")
            a.loc[0, "is_holdout"] = not bool(a.loc[0, "is_holdout"])
            a.to_csv(folder/"holdout_assignment.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Changed supplied split"):
                load_holdout(self.data, self.audit, folder)

    def test_reported_test_metrics_match_saved_predictions(self):
        rows = pd.read_csv(ROOT/"results/test_predictions.csv")
        scores = pd.read_csv(ROOT/"results/metrics.csv")
        expected = self.data.iloc[self.test_idx].set_index("excel_row")[TARGET]
        self.assertEqual(set(rows.excel_row), set(expected.index))
        truth = rows.excel_row.map(expected).to_numpy()
        np.testing.assert_allclose(truth, rows[TARGET].to_numpy(), atol=1e-12)
        for record in scores.to_dict("records"):
            error = rows[record["model"]].to_numpy() - truth
            self.assertAlmostEqual(float(np.abs(error).mean()), record["test_MAE"], places=10)
            self.assertAlmostEqual(float(np.sqrt((error**2).mean())), record["test_RMSE"], places=10)
            r2 = 1 - (error**2).sum()/((truth-truth.mean())**2).sum()
            self.assertAlmostEqual(float(r2), record["test_R2"], places=10)

    def test_predict_reorders_columns_and_rejects_invalid_inputs(self):
        bundle = joblib.load(ROOT/"results/models/selected.joblib")
        frame = pd.read_csv(ROOT/"examples/new_mixtures.csv")
        base = predict(frame, bundle).predicted_strength_mpa
        reordered = predict(frame[list(reversed(frame.columns))], bundle).predicted_strength_mpa
        np.testing.assert_allclose(base, reordered, rtol=0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "Missing prediction columns"):
            predict(frame.drop(columns=FEATURES[0]), bundle)
        broken = frame.copy()
        broken.loc[0, FEATURES[0]] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            predict(broken, bundle)
        broken = frame.copy()
        broken.loc[0, "age_days"] = 0
        with self.assertRaisesRegex(ValueError, "Age must be positive"):
            predict(broken, bundle)
        broken = frame.copy()
        broken.loc[0, FEATURES[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            predict(broken, bundle)


if __name__ == "__main__":
    unittest.main()
