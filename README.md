# Data error correction for software defect prediction: experiment code

This code reproduces Section 4 (Tables 2, 5–10, Fig. 2) of "An Empirical Study on Data Error Correction for Software Defect Prediction".

## Data
Use the PROMISE ARFF versions of NASA CM1, JM1, KC1 and PC1 (498 / 10,885 / 2,109 / 1,109 modules, 21 static code metrics plus a defect label). The run reported in the manuscript used the copies in
https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository (commit 3898d21e287c130c76d458faf65793d32483076d). Those copies come from the PROMISE site (http://promise.site.uottawa.ca/SERepository/datasets-page.html).

SHA-256 of each file used (also recorded in `manifest.json`):

| file | sha256 |
|---|---|
| cm1.arff | fbd4a74687d24350ee90b67211698c15406be0b6e435b98f5039e94302b97f09 |
| jm1.arff | 8ac4744e0807b49158964a05f1066da3a9e7f8fa27f47a52e80031e7942c5d97 |
| kc1.arff | 82fa208e6fc2c3b96072401c11c077897f713d8d93f7ad4cda89b8e4636bb5c0 |
| pc1.arff | b57312981180a748de4fc618e8c1173e01fc9d5c9923048e1aaa2d4572a95b35 |

Use the `.arff` files, not the `.csv` files. In that repository `jm1.csv` has 13,204 rows, but the PROMISE JM1 has 10,885.

## Run
```
pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost matplotlib joblib
python sdp_experiment.py --data DATA_DIR --out results --repeats 5 --folds 10
python make_figure.py results
python make_tables.py results      # manuscript-ready tables (Markdown + CSV)
```
For a quick timing check, first run with `--repeats 1`.

## What the pipeline does (per fold, fitted on the training partition only)
1. Common cleaning, done once per dataset before splitting and identically for all conditions. It removes records with a missing label and exact duplicate records (identical metrics and label), keeping one copy of each.
2. Approach-specific correction:
   - `none`: no correction.
   - `duplicate`: removes every training record whose metric vector also appears with the opposite label (inconsistent records).
   - `geo`: Golden Eagle Optimizer wrapper feature selection, Eqs. (3)–(8). Population 10, 20 iterations, sigmoid transfer function. Fitness is the inner 3-fold stratified-CV MCC of a decision tree, with SMOTE applied inside the inner folds, minus 0.01·|mask|/d. Attack propensity rises from 0.5 to 2; cruise propensity falls from 1 to 0.5.
   - `infotheory`: information gain, Eqs. (9)–(10), on 10 equal-frequency bins per metric. Keeps the top k, with k ∈ {5, 10, 15} chosen by inner 3-fold CV (decision tree, MCC).
3. Median imputation and z-score standardisation.
4. SMOTE (k = 5) to a 1:1 class ratio, on training data only.
5. Classifiers: DT (default settings), RF (300 trees), XGBoost (300 estimators), SVM (RBF kernel, default settings). Seeds derive from 42.
6. Test folds keep the natural defect rate. Metrics are computed for the defective class.

Statistics: Friedman test across the four conditions, then pairwise Wilcoxon signed-rank tests (Holm-corrected over the 6 pairs) and the Vargha–Delaney A12 effect size, within each dataset × classifier × metric.

## Outputs
`cleaning_log.json`, `fold_level_results.csv` (one row per fold × approach × classifier, with the confusion matrix), `fold_level_selection.csv` (selected metrics and records removed per fold), `table_mean_sd.csv`, `friedman_tests.csv`, `pairwise_tests.csv`, `manifest.json` (settings, seeds, versions, data hashes, run time), `fig2_mcc_auc.png/pdf`.
