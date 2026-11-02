# Evidence register: Okasha_CMC_revised_v02.docx

Every number added to the manuscript in this revision comes from output that `sdp_experiment.py` / `sdp_pca_ablation.py` produced in this session (label **R**, a result), from arithmetic on those outputs (**R-derived**), or from the dataset files themselves (**D**, data). Interpretations added to the text are written as hypotheses ("a possible explanation, not tested here").

The text was inserted by a script that reads the CSV files listed below, so no value was typed in by hand. Afterwards, every cell of Tables 6–10 was parsed back out of the .docx and compared with an independent recomputation from `fold_level_results.csv`: 304 cells checked, 0 mismatches.

Run: 5 × 10 repeated stratified CV, seed 42, 200 dataset-folds, 28.9 min on 2 vCPUs (`manifest.json`).

## Numbers by manuscript location

| Location | Statement / number(s) | Label | Source |
|---|---|---|---|
| Abstract | No approach met A12 ≥ 0.64 and \|ΔMCC\| ≥ 0.05 as an increase | R | `pairwise_tests.csv` (metric=mcc, reference=none) |
| Abstract, §5.1, §6 | Duplicate-bug-report ΔMCC −0.027 … +0.013 | R | `pairwise_tests.csv` mean_diff, treatment=duplicate |
| Abstract, §4.2, §4.4, §6 | 12 Holm-significant MCC differences, all decreases (6 GEO, 6 information-theory) | R | `pairwise_tests.csv` p_holm < 0.05 |
| Abstract, §4.4, §5.1, §6 | KC1 / XGBoost / information-theory: ΔMCC −0.076, A12 0.28 (the only change meeting both practical thresholds) | R | same |
| §3.1, Table 2 | 498/49, 10,885/2,106, 2,109/326, 1,109/77; 21 metrics | D | ARFF files (SHA-256 in `manifest.json`) |
| Table 2 | Defective %: CM1 9.84 (draft had 9.83), KC1 15.46 (draft had 15.45) | D-derived | 49/498, 326/2,109 |
| Table 2, §3.1 | After cleaning: 442 (48; 10.86%), 8,912 (2,007; 22.52%), 1,212 (315; 25.99%), 954 (70; 7.34%) | R | `cleaning_log.json` |
| §3.2, §5.2 | Exact duplicates removed: 56 / 1,973 / 897 / 155 (11.2 / 18.1 / 42.5 / 14.0%) | R | `cleaning_log.json` |
| §3.2.1 | JM1: 25 missing values in 5 modules; other datasets none | D | `cleaning_log.json` |
| §3.3.1, §5.1 | Records removed per training fold: 1.6 (0.41%), 142.6 (1.78%), 32.5 (2.98%), 9.6 (1.12%) | R | `fold_level_selection.csv` n_train_removed ÷ n_train |
| §3.3.2 | GEO settings: pop 10, T 20, 3 inner folds, α 0.01, p_a 0.5→2, p_c 1→0.5, bounds ±6 | Setting (rerun) | `manifest.json` → geo. **Author to confirm** these match the original GEO runs |
| §3.3.2 | 40.6 distinct masks per run; 10.2–11.1 metrics retained | R | `fold_level_selection.csv` |
| §3.3.3 | 10 equal-frequency bins; k ∈ {5, 10, 15} by inner CV; mean k 8.3 / 10.7 / 11.8 / 12.9 | Setting + R | `manifest.json`, `fold_level_selection.csv` |
| Table 4 | Python 3.13.18, scikit-learn 1.9.1, imbalanced-learn 0.15.dev0, xgboost 2.1.4 | R | `manifest.json` → environment |
| §3.6 | 28.9 min; GEO 0.75 / 7.79 / 1.20 / 1.16 s per fold | R | `manifest.json`, `fold_level_selection.csv` |
| §3.7, §4.1, Table 5 | 7.34–25.99% defective; 74.01–92.66% trivial accuracy; F1 = 2π/(1+π): 0.196 / 0.368 / 0.413 / 0.137 | R-derived | `cleaning_log.json` defect_rate_clean |
| Tables 6–9, S1 | Means ± SD over 50 folds | R | `table_mean_sd.csv` |
| Table 10 | ΔMCC (A12), * where p_Holm < 0.05 | R | `table10.csv`, `pairwise_tests.csv` |
| §4.2 | Largest increases: IT/DT/CM1 +0.058 (A12 0.60, p 0.075); IT/RF/CM1 +0.039 (0.57, p 1.000); best MCC per dataset 0.273 / 0.258 / 0.267 / 0.351; best AUC 0.676–0.841 | R | `pairwise_tests.csv`, `table_mean_sd.csv` |
| §4.3 | Recall changes +0.041 (IT, SVM, KC1) and −0.043 (dup, SVM, PC1); SVM recall 0.49–0.73 at precision 0.19–0.43; other classifiers 0.17–0.43; F1 and MCC pick the same top condition in 15/16 | R | same |
| §4.4 | Ranked first by MCC: none 4, duplicate 8, GEO 2, IT 2; 19 Holm-significant AUC decreases, 0 increases | R | same |
| §4.4 | PCA ablation: 7.0–8.0 components; ΔMCC −0.096 … +0.067; ΔAUC −0.040 … +0.031; 8/16 MCC differences significant, all decreases | R | `pca_ablation/pca_ablation_summary.csv` (S3) |
| §4.5 | Max F1 from rounded P = 0.55, R = 0.51 is 0.534 (draft figure shows 0.54) | Arithmetic on the author's Fig. 3 values | 2·0.555·0.515/(1.07) |
| §4.5 | Highest rerun F1 0.353 / 0.445 / 0.462 / 0.391 | R | `table_mean_sd.csv` |
| §5.1 | KC1 MCC > CM1 MCC by > 0.02 for DT, RF, XGBoost (no correction) | R | `table_mean_sd.csv` |
| §5.2 | KC1 defect rate 15.46% → 25.99% after cleaning | D / R | `cleaning_log.json` |

## Statements that are author decisions, not evidence (highlighted in the .docx)
- Operational definition of the "duplicate-bug-report-based" approach (inconsistent-record removal on metric data). I chose this definition; you have not confirmed it matches your original method.
- The GEO and information-gain settings, and dropping PCA from the main pipeline (it is now an ablation only).
- Practical-significance thresholds (A12 ≥ 0.64, |ΔMCC| ≥ 0.05). They come from the previous draft; please confirm they were fixed before the results were seen.
- The data version (PROMISE ARFF from the GitHub mirror, commit 3898d21) is the same as in your original experiments.
- Details of the earlier single-split runs: split ratio, classifier, preprocessing order, and whether the information-theory runs were separate.

## Not done in this revision
- References were **not** verified (Stage 3). No citation was added, removed or changed. The literature gap in §1 is tagged for confirmation.
- Back-matter facts (funding, acknowledgement, conflicts of interest, author list, code DOI) still need author input.
- The rerun used an independent implementation in this session, not your original code. Numbers from your original runs (Fig. 3) remain as reported, with flagged inconsistencies.
- imbalanced-learn and XGBoost were built from GitHub source because PyPI was unreachable from this workspace. If you rerun with released versions, small numeric differences are possible.
