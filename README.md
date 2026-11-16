# Geopolymer concrete compressive strength prediction

CV7001 mini-project, ZHANG YUNHAN (G2509929H).

Repository: <https://github.com/yhzhangchn/CV7001-ML>.

This project uses Ridge regression, Random Forest and XGBoost to predict compressive strength in MPa from mixture composition, curing conditions and specimen age.

## Data and inputs

The data come from [Aravind-Unni/GPC-Dataset](https://github.com/Aravind-Unni/GPC-Dataset). The local 820-record copy is included as `data/gpc_database_820.csv`, ready for training. `prepare_data.py` extracted it from the previously downloaded `Raw_GitHub_820` worksheet. The source workbook hash and column mapping are recorded in `data/source_manifest.json`.

The model uses **10 of the 13 supplied predictors**:

- Seven material amounts: NaOH, sodium silicate, GGBFS, fly ash, extra water, fine aggregate and coarse aggregate (kg/m³).
- NaOH molarity (mol/L), curing temperature (°C) and specimen age (days).

Exact field names and source notes are in [the data dictionary](data/DATA_DICTIONARY.md).

## Run the project

Tested with Python 3.12.14 on Windows. From this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py
.\.venv\Scripts\python.exe analyze_results.py
.\.venv\Scripts\python.exe verify_reproduction.py
```

If an environment is already installed, use the last four commands. On Linux/macOS, create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python`. `requirements.txt` pins the package versions used for the supplied results.

## Evaluation and results

The seven material amounts define 455 base-composition groups, rounded to eight decimals. Records of the same composition stay together across ages, molarities and curing temperatures. A fixed grouped split (seed 42) assigns 670 records/364 groups to development and 150 records/91 groups to testing. The test fraction is 20% of groups and 18.3% of records.

Five-fold shuffled GroupKFold evaluates the development set, also with seed 42. Ridge scaling is fitted in each training fold. The three models use fixed settings and identical folds. The lowest mean CV RMSE selects the model for inference; test scores are then calculated on the held-out records.

Ridge uses alpha 1. Random Forest uses 300 trees and maximum depth 8. XGBoost uses 300 trees, maximum depth 6, learning rate 0.05, subsample 0.8 and colsample_bytree 0.8. Both tree models use seed 42 and two CPU workers. The permutation analysis uses ten repeats per feature and seed 42 + fold number (folds 1-5).

| Model | CV RMSE (mean ± SD) | Test R² | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| Ridge | 11.87 ± 1.35 | 0.084 | 12.98 | 9.67 |
| Random Forest | 7.12 ± 1.12 | 0.713 | 7.27 | 4.67 |
| XGBoost | 7.00 ± 1.10 | 0.740 | 6.92 | 4.29 |

Errors are in MPa. XGBoost reduces test RMSE by 46.7% relative to Ridge and 4.8% relative to Random Forest. Of its 150 test predictions, 109 (72.7%) are within 5 MPa and 134 (89.3%) within 10 MPa.

| Measured strength | Test records | MAE (MPa) | Mean prediction error (MPa) |
| --- | --- | --- | --- |
| Below 20 MPa | 10 | 5.31 | +5.16 |
| 20 to below 60 MPa | 129 | 3.67 | +0.77 |
| 60 MPa and above | 11 | 10.69 | -7.95 |

Errors are smallest in the middle range. The model tends to overpredict lower strengths and underpredict higher strengths. The high-strength range has only 52 development records, making it a priority for expanding the training data.

Development-fold permutation analysis ranks GGBFS content, NaOH molarity and curing temperature highest. Shuffling these inputs increases validation RMSE by 7.52, 7.16 and 2.52 MPa, respectively (mean over five folds, ten permutations per input). This measures the selected model's reliance on each input. The main data limitations are zero-coded molarities and missing study labels; details are in the data dictionary.

`train.py` writes full scores, predictions, split assignments, the figure and fitted models to `results/`. To save another run separately:

```bash
python train.py --output-dir alternative_results
```

## Predict from a CSV

```bash
python predict.py --input examples/new_mixtures.csv --output results/example_predictions.csv
```

The example file demonstrates one composition at 7 and 28 days. New files use the ten field names in the data dictionary; column order can vary and additional identifiers are preserved. The script checks the values and flags features outside the training range.

To use the fitted Random Forest:

```bash
python predict.py --model results/models/random_forest.joblib --input examples/new_mixtures.csv --output results/rf_examples.csv
```

## Reproduce the results

```bash
python verify_reproduction.py
```

This runs six checks for the data/split, grouping, fold-local scaling, split-file integrity, reported scores and prediction inputs. It then copies code, data and examples to an empty folder, trains fresh models and compares metrics, assignments, predictions and analysis tables with the saved results. The tolerance is 1e-10 in the same Python environment. The record is `results/reproducibility_check.json`.

`analyze_results.py` produces the strength-band errors, error thresholds and development-fold permutation scores. Outputs are `results/strength_bands.csv`, `error_coverage.csv`, `development_permutation_importance.csv`, `permutation_folds.csv` and `analysis_summary.json`.

The included CSV is sufficient for these commands. To repeat extraction from the existing collection workbook:

```bash
python prepare_data.py --workbook "PATH/TO/geopolymer_broad_collection_886_rows.xlsx" --output-dir extracted_data
```

## Report

The two-page report is `report/main.pdf`; its source is `report/main.tex`. The repository URL is stored in `report/repository_url.tex`. From `report/`, compile with `tectonic main.tex` or run `pdflatex main.tex` twice.

After changing a training run, copy `results/metrics.tex`, `results/heldout_predictions.png`, `results/heldout_predictions.pdf` and `results/source_index.csv` to their matching paths under `report/`, then recompile. The report uses the vector PDF figure. The plot uses Times New Roman where installed, with STIXGeneral as the fallback.

## References and AI assistance

Data: Aravind-Unni, *GPC Dataset*, linked above. Methods: Pedregosa et al., *JMLR* 12 (2011), 2825-2830; Chen and Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD (2016), DOI 10.1145/2939672.2939785.

Generative AI assisted with code editing, verification and report drafting.
