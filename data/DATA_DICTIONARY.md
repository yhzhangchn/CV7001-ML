# Data dictionary

`gpc_database_820.csv` contains the local 820-row copy of [Aravind-Unni/GPC-Dataset](https://github.com/Aravind-Unni/GPC-Dataset), extracted from the existing collection workbook's `Raw_GitHub_820` sheet. It preserves all 13 source predictors and the strength target. `source_manifest.json` records the original headers, mapping and file hashes.

## Model inputs

| CSV field | Source field | Unit |
| --- | --- | --- |
| `naoh_content_kg_m3` | NaOH Content (SH) | kg/m³ |
| `sodium_silicate_kg_m3` | Na2SiO3 (SS) | kg/m³ |
| `ggbfs_kg_m3` | GGBFS | kg/m³ |
| `fly_ash_kg_m3` | Fly Ash (FA) | kg/m³ |
| `extra_water_kg_m3` | Extra Water (EW) | kg/m³ |
| `fine_aggregate_kg_m3` | Fine Aggregates (F) | kg/m³ |
| `coarse_aggregate_kg_m3` | Coarse aggregates (CA) | kg/m³ |
| `naoh_molarity_mol_l` | Molarity of NaOH (M) | mol/L |
| `curing_temperature_c` | Curing Temperature (T) | °C |
| `age_days` | Age of Mix (Days) | days |

The target is `strength_mpa`, measured compressive strength in MPa. Age is specimen age. NaOH content follows the source's reported quantity; its solution/solid basis across studies is unspecified. Extra water is the additional-water field.

The three unused predictors are `ss_sh_ratio`, `alkaline_binder_ratio` and `superplasticizer_kg_m3`. They remain in the snapshot. The model's ten inputs were selected before training.

## Rows and data quality

All 820 rows have numerical fields and positive strength and age. They are retained, including repeated selected input vectors. There are 26 zero entries for NaOH molarity whose coding needs verification; source values are preserved. The public collection lists its literature sources but has no per-record study labels.

`record_id` runs from 0 to 819. Exported `sample_id` values GPC0001-GPC0820 identify these rows, and `excel_row` records rows 2-821 of the source worksheet. `source_id=GPC820` identifies the whole collection.

## Groups and split

The seven material amounts, rounded to eight decimals, define 455 base-composition groups. Ages and processing variants of a base composition stay together. `GroupShuffleSplit(test_size=0.2, random_state=42)` gives 670 development rows/364 groups and 150 test rows/91 groups. Five-fold grouped CV uses the development rows, with seed 42.
