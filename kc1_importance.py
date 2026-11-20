#!/usr/bin/env python3
"""
Reviewer 1, Major 6: why does information-gain selection lower MCC for XGBoost on KC1?
On the same 50 outer folds and seeds as the main run, fit the no-correction pipeline
(all 21 metrics; median imputation, standardisation, SMOTE on the training fold only)
with XGBoost (300 trees) and measure permutation importance on each untouched test
fold (MCC and ROC-AUC, 5 permutations per metric). The output is compared with how
often IG dropped each metric on KC1 (selection_main.csv).
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, matthews_corrcoef
import sdp_experiment_v2 as V


def fold_importance(X, y, names, rep, fold, tr, te):
    seed = V.SEED + 1000 * rep + fold
    prep = V.Prep().fit(X[tr])
    Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
    Xs, ys = V.smote(Xtr, y[tr], seed)
    clf = V.make_classifiers(seed)["XGBoost"].fit(Xs, ys)
    rows = []
    for sc_name, scorer in [("mcc", make_scorer(matthews_corrcoef)), ("auc", "roc_auc")]:
        r = permutation_importance(clf, Xte, y[te], scoring=scorer, n_repeats=5, random_state=seed, n_jobs=1)
        for n, m, s in zip(names, r.importances_mean, r.importances_std):
            rows.append(dict(dataset="KC1", repeat=rep, fold=fold, metric_scored=sc_name, feature=n,
                             importance_mean=m, importance_sd=s))
    return rows


def run(data_dir, n_jobs=6):
    from joblib import Parallel, delayed
    data, logs, tasks = V.build_tasks(data_dir, V.Config(), datasets=["KC1"])
    X, y, names = data["KC1"]
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fold_importance)(X, y, names, rep, f, tr, te)
                                                  for _, rep, f, tr, te in tasks)
    return pd.DataFrame([r for rr in res for r in rr])


if __name__ == "__main__":
    # python kc1_importance.py DATA_DIR OUT_DIR [N_JOBS]
    import os, sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results_rev1"
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    os.makedirs(out_dir, exist_ok=True)
    df = run(data_dir, n_jobs=n_jobs)
    df.to_csv(f"{out_dir}/kc1_permutation_importance.csv", index=False)
    print(df.shape, "written to", f"{out_dir}/kc1_permutation_importance.csv")
