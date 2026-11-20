# Data-quality interventions for software defect prediction: experiment code (revision 1)

This repository holds the code and outputs for the revised manuscript submitted to
Computers, Materials & Continua (submission 96252). The study compares three
interventions against a common no-intervention baseline on the NASA MDP datasets
CM1, JM1, KC1 and PC1: inconsistent-record removal (instance filtering), Golden Eagle
Optimizer (GEO) wrapper feature selection, and information-gain (IG) filter selection.
Four classifiers are used: decision tree (DT), random forest (RF, 300 trees),
XGBoost (300 trees) and an RBF support vector machine (SVM).

## What changed in revision 1

1. **Nested IG selection.** In the previous version, IG ranked the metrics and chose k
   on the whole outer training fold, and the preprocessing used inside that choice was
   not refitted per inner split. IG ranking, discretisation, imputation and scaling are
   now fitted inside each inner training split. The final ranking uses the outer
   training fold only. The outer test fold is never used for selection.
2. **GEO preprocessing per inner split.** GEO did not use labels outside the outer
   training fold in the previous version. Its unsupervised preprocessing is now also
   refitted per inner split, for consistency.
3. **Corrected confidence intervals.** Differences from the no-intervention baseline
   are reported with the corrected repeated-CV t interval (Nadeau and Bengio 2003;
   Bouckaert and Frank 2004). This interval is an approximation for repeated k-fold CV.
4. **Sensitivity runs.** Duplicates kept; IG with 5 or 20 bins; IG with fixed k = 5, 10
   or 15; GEO with 20 agents and 50 iterations; selection tuned with the downstream
   classifier (inner RF or XGBoost with 100 trees) instead of an inner DT.
5. **Reference run of the previous IG code path** (`v03ref_ig`), on the same folds and
   in the same environment, to isolate the effect of the nesting fix.
6. **Feature-selection stability** (Nogueira, Sechidis and Brown 2018), training-set MCC
   for an overfitting check, and a KC1 permutation-importance analysis.
7. The condition formerly called "duplicate-bug-report-based" is renamed
   **inconsistent-record removal**. It uses only the 21 static code metrics. No bug-report
   text is used anywhere in this study.

## Files

| file | purpose |
|---|---|
| `sdp_experiment_v2.py` | data loading, cleaning, the four conditions, classifiers, one outer fold (`run_fold`) |
| `run_experiments.py` | runs the named configurations and writes `folds_`, `selection_` and `run_` files |
| `analysis_rev1.py` | effect sizes, corrected CIs, A12, thresholds, stability, RQ2 decomposition, overfitting |
| `make_tables_rev1.py` | writes every analysis table to `results_rev1/tables/` |
| `make_figures_rev1.py` | draws the main forest plot and Supplementary Figures S1–S3 |
| `kc1_importance.py` | permutation importance of all 21 metrics for XGBoost on KC1 |
| `overlap_check.py` | share of test modules with an identical metric vector in the training fold |
| `verify_rerun.py` | compares a rerun with the reference outputs |
| `check_headlines.py` | prints the headline numbers listed below |
| `manifest_rev1.json` | versions, seeds, data SHA-256, run times and SHA-256 of every output file |
| `requirements_rev1.txt` | exact package versions used for the reference outputs |
| `results_rev1/` | reference outputs of revision 1 |
| `sdp_experiment.py`, `make_tables.py`, `make_figure.py`, `sdp_pca_ablation.py`, `results/` | previous version, kept unchanged for comparison |

## Data

Use the PROMISE ARFF files in `data/`. They come from
https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository (commit
3898d21e287c130c76d458faf65793d32483076d). Use the `.arff` files, not the `.csv` files.
The SHA-256 of each file is recorded in `manifest_rev1.json`, and every `run_*.json`
records the hash of the file it actually read.

| dataset | modules | defective | after exact-duplicate removal (defective) | SHA-256 (first 12) |
|---|---|---|---|---|
| CM1 | 498 | 49 | 442 (48) | fbd4a74687d2 |
| JM1 | 10,885 | 2,106 | 8,912 (2,007) | 8ac4744e0807 |
| KC1 | 2,109 | 326 | 1,212 (315) | 82fa208e6fc2 |
| PC1 | 1,109 | 77 | 954 (70) | b57312981180 |

## Protocol

5 × 10 repeated stratified cross-validation (`RepeatedStratifiedKFold`, `random_state=42`).
Each fold uses seed 42 + 1000 × repeat + fold for SMOTE, the classifiers and GEO. SMOTE
(k = 5) is applied to the training fold only. Exact duplicate rows are removed once per
dataset before splitting, identically for all conditions (configuration `nodedup` keeps
them). GEO and IG are tuned with three inner folds of the outer training fold.

## Verification guide

The reference outputs were produced on Windows 11 with Python 3.11.18 and the versions in
`requirements_rev1.txt`. With the same versions, a rerun is expected to agree exactly.
This was checked on three folds (48 result rows): the largest absolute difference in MCC,
AUC, F1 and training MCC was 0.0.

### Step 1. Install (about 5 minutes)

```
python -m venv .venv
.venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
pip install -r requirements_rev1.txt
```

### Step 2. Quick check (about 1 minute)

```
python run_experiments.py --data data --out quickcheck --quick --n_jobs 1
python verify_rerun.py --rerun quickcheck
```

The report should say `EXACT` for `main` (48 rows). `WITHIN` means small numerical
differences, usually from other library versions; continue to Step 3 and compare the
headline numbers. `DIFFERENT` means that something in the setup differs; check the
data hashes in `quickcheck/run_main.json` against `manifest_rev1.json`.

### Step 3. Main run (82 minutes with 6 parallel jobs on 8 CPUs)

```
python run_experiments.py --data data --out rerun --configs main --n_jobs 6
python verify_rerun.py --rerun rerun
```

Run time scales roughly with the number of parallel jobs. If parallel jobs fail on a
restricted machine, use `--n_jobs 1`; this is slower but gives the same numbers.

### Step 4. All configurations (282 minutes in total with 6 jobs; optional)

```
python run_experiments.py --data data --out rerun --configs all --n_jobs 6
python kc1_importance.py data rerun          (writes rerun/kc1_permutation_importance.csv)
python overlap_check.py data rerun/tables/train_test_vector_overlap.csv
python verify_rerun.py --rerun rerun
python make_tables_rev1.py --results rerun
python make_figures_rev1.py --tables rerun/tables --out rerun/figures
python check_headlines.py --tables rerun/tables
```

Run times of the reference outputs (minutes, 6 jobs): geo_p20_i50 54.2, ig_bins20 7.6, ig_bins5 7.3, ig_k10 6.6, ig_k15 7.1, ig_k5 5.1, main 82.0, matched_geo_XGB 37.3, matched_ig_RF 18.6, matched_ig_XGB 3.7, nodedup 23.9, nodedup_svm_none 11.4, v03ref_ig 16.1;
KC1 importance 1.1.

### Step 5. Compare the headline numbers

`check_headlines.py` on the reference tables prints the lines below. A rerun with the
same versions should print identical lines. With other versions, a difference of up to
0.005 in a mean or a CI bound is within tolerance, but a change in any count (for example,
the number of CIs above zero) should be reported to the authors.

```
H1  MCC cells: 48; CI above 0: 0; CI below 0: 3
    below 0: geo/JM1/RF -0.027 [-0.047, -0.007]
    below 0: infotheory/KC1/RF -0.062 [-0.122, -0.002]
    below 0: infotheory/KC1/XGBoost -0.074 [-0.141, -0.006]
H2  AUC cells: CI above 0: 0; CI below 0: 7
H3  threshold rule (A12 >= 0.64 and dMCC >= 0.05): increases 0 decreases 2
H4  dMCC range by approach: {'geo': (-0.079, 0.043), 'inconsistent': (-0.028, 0.011), 'infotheory': (-0.074, 0.052)}
H5  best mean MCC per dataset: {'CM1': 0.273, 'JM1': 0.258, 'KC1': 0.259, 'PC1': 0.355}
H6  Nogueira stability: {'geo/CM1': -0.011, 'infotheory/CM1': 0.51, 'geo/JM1': 0.027, 'infotheory/JM1': 0.555, 'geo/KC1': 0.016, 'infotheory/KC1': 0.627, 'geo/PC1': 0.02, 'infotheory/PC1': 0.635}
H7  nesting-fix effect on IG dMCC: -0.022 to +0.013 (mean -0.002)
H8  selector settings, CI above 0 per setting: {'geo_main_p10_i20': 0, 'geo_p20_i50': 0, 'ig_bins20': 0, 'ig_bins5': 0, 'ig_k10': 0, 'ig_k15': 0, 'ig_k5': 0, 'ig_main_bins10_kselected': 0}
H9  matched selection: cells 12; CI above 0 0; CI below 0 0
H10 duplicates kept: cells 36; CI above 0 0; CI below 0 1
```

The labels mean: `none` no intervention, `inconsistent` inconsistent-record removal,
`geo` GEO wrapper selection, `infotheory` information-gain selection.

## Known limitations

- The corrected t interval is an approximation for repeated k-fold CV; p-values and
  intervals are descriptive.
- The practical thresholds A12 ≥ 0.64 and |ΔMCC| ≥ 0.05 were entered after the results of
  the previous version existed. They are the conventional Vargha–Delaney medium effect and
  the MCC margin of the earlier draft. `threshold_grid.csv` gives the counts for other
  threshold pairs.
- The classifier-matched GEO run was done for XGBoost only, and the selector-setting runs
  use DT, RF and XGBoost (not SVM), to limit compute.
- Four NASA datasets with static code metrics only; results may not transfer to other
  metric sets or projects.
