#!/usr/bin/env python3
"""Supplementary ablation: does PCA (95% training-fold variance, fitted after
standardisation, before SMOTE) change performance of the no-correction
pipeline? Uses exactly the same folds, seeds and classifiers as
sdp_experiment.py.
Usage: python sdp_pca_ablation.py --data DATA_DIR --out OUT_DIR"""
import argparse, os, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
import sdp_experiment as E

PCA_VAR = 0.95


def fold(ds, X, y, rep, k, tr, te):
    seed = E.SEED + 1000 * rep + k
    prep = E.Prep().fit(X[tr])
    Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
    rows = []
    for cond in ["none", "none+PCA"]:
        A, B = Xtr, Xte
        ncomp = A.shape[1]
        if cond == "none+PCA":
            p = PCA(n_components=PCA_VAR, svd_solver="full", random_state=seed).fit(A)
            A, B, ncomp = p.transform(A), p.transform(B), p.n_components_
        Xs, ys = E.smote(A, y[tr], seed)
        for c, clf in E.make_classifiers(seed).items():
            clf.fit(Xs, ys)
            m = E.metrics(y[te], clf.predict(B), E.scores_for(clf, B))
            rows.append(dict(dataset=ds, repeat=rep, fold=k, condition=cond,
                             classifier=c, n_components=int(ncomp), **m))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = []
    for ds in E.DATASETS:
        Xdf, y = E.load_arff(os.path.join(a.data, ds.lower() + ".arff"))
        X, yv, _, _ = E.common_cleaning(Xdf, y)
        r = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats,
                                    random_state=E.SEED)
        for i, (tr, te) in enumerate(r.split(X, yv)):
            tasks.append((ds, X, yv, i // a.folds, i % a.folds, tr, te))
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(fold)(*t) for t in tasks)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(a.out, "pca_ablation_folds.csv"), index=False)
    out = []
    for (ds, c), g in df.groupby(["dataset", "classifier"]):
        for m in ["mcc", "auc", "recall", "f1"]:
            w = g.pivot_table(index=["repeat", "fold"], columns="condition", values=m)
            d = w["none+PCA"] - w["none"]
            p = stats.wilcoxon(w["none+PCA"], w["none"], zero_method="zsplit").pvalue \
                if np.any(d != 0) else 1.0
            out.append(dict(dataset=ds, classifier=c, metric=m,
                            mean_none=w["none"].mean(), sd_none=w["none"].std(),
                            mean_pca=w["none+PCA"].mean(), sd_pca=w["none+PCA"].std(),
                            mean_diff=d.mean(), A12=E.a12(w["none+PCA"], w["none"]),
                            p_raw=p,
                            n_components_mean=g[g.condition == "none+PCA"].n_components.mean()))
    s = pd.DataFrame(out)
    # Holm over the 4 metrics x 4 classifiers within each dataset
    s["p_holm"] = s.groupby("dataset")["p_raw"].transform(lambda p: E.holm(p.values))
    s.to_csv(os.path.join(a.out, "pca_ablation_summary.csv"), index=False)
    print("done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
