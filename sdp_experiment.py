#!/usr/bin/env python3
"""
Leakage-controlled comparison of data error correction approaches for
software defect prediction on the PROMISE versions of NASA CM1, JM1, KC1, PC1.

Conditions (one per run, all on identical folds):
  none        - common cleaning + imputation + scaling + SMOTE, no correction
  duplicate   - instance-level: in the TRAINING fold, remove every record whose
                metric vector also occurs with the opposite label
                (inconsistent records, Shepperd et al. 2013 cleaning rule)
  geo         - feature-level wrapper: Golden Eagle Optimizer (GEO) feature mask,
                fitness = mean inner-CV MCC - alpha * |mask| / d   (Eq. 8)
  infotheory  - feature-level filter: entropy / information gain (Eqs. 9-10)
                on equal-frequency discretised metrics, top-k with k chosen
                by inner CV from K_GRID

Every data-dependent step (median imputation, standardisation, correction,
SMOTE) is fitted on the training partition of the fold only. Test partitions
keep the natural defect rate.

Usage:
  python sdp_experiment.py --data DATA_DIR --out OUT_DIR [--repeats 5 --folds 10]
"""
import argparse, hashlib, io, json, os, platform, re, sys, time
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, roc_auc_score)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False

DATASETS = ["CM1", "JM1", "KC1", "PC1"]
APPROACHES = ["none", "duplicate", "geo", "infotheory"]
SEED = 42

# ---- approach settings (reported in the manuscript, Section 3.3) ----------
GEO = dict(pop=10, iters=20, inner_folds=3, alpha=0.01,
           pa0=0.5, paT=2.0, pc0=1.0, pcT=0.5, bound=6.0)
IG = dict(bins=10, k_grid=(5, 10, 15), inner_folds=3)
SMOTE_K = 5


# ---------------------------------------------------------------- data ------
def load_arff(path):
    lines = open(path, errors="replace").read().splitlines()
    attrs = [re.split(r"\s+", l.strip())[1] for l in lines
             if l.lower().startswith("@attribute")]
    start = [i for i, l in enumerate(lines) if l.lower().startswith("@data")][0]
    rows = [l for l in lines[start + 1:] if l.strip() and not l.startswith("%")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=attrs,
                     na_values=["?"])
    y = df.iloc[:, -1].astype(str).str.strip().str.lower()
    y = y.map({"true": 1, "yes": 1, "false": 0, "no": 0})
    X = df.iloc[:, :-1].astype(float)
    return X, y


def common_cleaning(X, y):
    """Applied once per dataset, identically for all conditions (Sec. 3.2)."""
    log = {"modules_raw": int(len(y)), "defective_raw": int(y.sum())}
    keep = y.notna()
    X, y = X[keep], y[keep].astype(int)
    log["removed_missing_label"] = int((~keep).sum())
    df = X.copy()
    df["__y"] = y.values
    dup = df.duplicated(keep="first")
    log["removed_exact_duplicates"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    X, y = df.drop(columns="__y"), df["__y"].astype(int)
    log["modules_clean"] = int(len(y))
    log["defective_clean"] = int(y.sum())
    log["defect_rate_clean"] = float(y.mean())
    log["missing_metric_cells"] = int(X.isna().sum().sum())
    log["rows_with_missing_metric"] = int(X.isna().any(axis=1).sum())
    keycols = list(X.columns)
    g = df.groupby(keycols, dropna=False)["__y"].transform("nunique")
    log["inconsistent_records_after_cleaning"] = int((g > 1).sum())
    log["n_metrics"] = int(X.shape[1])
    return X.values, y.values, list(X.columns), log


# ------------------------------------------------------- preprocessing ------
class Prep:
    """Median imputation (Eq. 1) + standardisation (Eq. 2), fitted on train."""
    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        Xi = np.where(np.isnan(X), self.med, X)
        self.sc = StandardScaler().fit(Xi)
        return self

    def transform(self, X):
        return self.sc.transform(np.where(np.isnan(X), self.med, X))


def smote(X, y, seed):
    k = min(SMOTE_K, int(y.sum()) - 1)
    if k < 1:
        return X, y
    return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)


def inner_dt_mcc(X, y, cols, seed, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(X, y):
        Xtr, ytr = smote(X[tr][:, cols], y[tr], seed)
        clf = DecisionTreeClassifier(random_state=seed).fit(Xtr, ytr)
        scores.append(matthews_corrcoef(y[va], clf.predict(X[va][:, cols])))
    return float(np.mean(scores))


# --------------------------------------------------------- approaches -------
def duplicate_correction(X_raw, y):
    """Remove all training records whose metric vector carries both labels."""
    df = pd.DataFrame(X_raw).fillna(-1e300)
    key = pd.util.hash_pandas_object(df, index=False).values
    s = pd.DataFrame({"k": key, "y": y})
    n_lab = s.groupby("k")["y"].transform("nunique").values
    keep = n_lab == 1
    return keep


def geo_select(X, y, seed):
    """GEO wrapper (Mohammadi-Balani et al., 2021), Eqs. (3)-(8)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    P, T, B = GEO["pop"], GEO["iters"], GEO["bound"]
    cache = {}

    def mask_of(pos):
        m = (1.0 / (1.0 + np.exp(-pos))) > 0.5
        if not m.any():
            m[np.argmax(pos)] = True
        return m

    def fitness(pos):
        m = mask_of(pos)
        key = m.tobytes()
        if key not in cache:
            cols = np.flatnonzero(m)
            cache[key] = (inner_dt_mcc(X, y, cols, seed, GEO["inner_folds"])
                          - GEO["alpha"] * m.sum() / d)
        return cache[key]

    pos = rng.uniform(-B, B, size=(P, d))
    fit = np.array([fitness(p) for p in pos])
    mem, memfit = pos.copy(), fit.copy()
    for t in range(1, T + 1):
        pa = GEO["pa0"] + (t / T) * (GEO["paT"] - GEO["pa0"])        # Eq. 6
        pc = GEO["pc0"] - (t / T) * (GEO["pc0"] - GEO["pcT"])
        for i in range(P):
            f = rng.integers(P)
            A = mem[f] - pos[i]                                       # Eq. 3
            nA = np.linalg.norm(A)
            if nA < 1e-12:
                continue
            r = rng.normal(size=d)
            C = r - (r @ A) / (nA ** 2) * A                           # Eq. 4
            nC = np.linalg.norm(C)
            step = rng.random() * pa * A / nA                         # Eq. 5
            if nC > 1e-12:
                step = step + rng.random() * pc * C / nC
            pos[i] = np.clip(pos[i] + step, -B, B)
            fi = fitness(pos[i])
            if fi > memfit[i]:
                mem[i], memfit[i] = pos[i].copy(), fi
    best = int(np.argmax(memfit))
    return np.flatnonzero(mask_of(mem[best])), int(len(cache))


def entropy(labels):
    p = np.bincount(labels, minlength=2) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def info_gain(X, y, bins):
    """IG(S,A) = H(S) - sum_v |S_v|/|S| H(S_v) on equal-frequency bins."""
    H = entropy(y)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
        v = np.searchsorted(edges, X[:, j], side="right")
        cond = 0.0
        for val in np.unique(v):
            sel = v == val
            cond += sel.mean() * entropy(y[sel])
        out[j] = H - cond
    return out


def ig_select(X, y, seed):
    ig = info_gain(X, y, IG["bins"])
    order = np.argsort(-ig, kind="stable")
    best_k, best_s = None, -np.inf
    for k in IG["k_grid"]:
        k = min(k, X.shape[1])
        s = inner_dt_mcc(X, y, order[:k], seed, IG["inner_folds"])
        if s > best_s:
            best_k, best_s = k, s
    return np.sort(order[:best_k]), ig


# --------------------------------------------------------- classifiers ------
def make_classifiers(seed):
    clfs = {
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "SVM": SVC(kernel="rbf", random_state=seed),
    }
    if HAVE_XGB:
        clfs["XGBoost"] = XGBClassifier(n_estimators=300, objective="binary:logistic",
                                        eval_metric="logloss", random_state=seed,
                                        n_jobs=1, tree_method="hist")
    return clfs


def scores_for(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def metrics(y, yhat, score):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
                precision=prec, recall=rec, f1=f1,
                mcc=matthews_corrcoef(y, yhat), auc=roc_auc_score(y, score),
                accuracy=(tp + tn) / len(y))


# --------------------------------------------------------------- fold -------
def run_fold(ds, X, y, names, rep, fold, tr, te):
    seed = SEED + 1000 * rep + fold
    rows, sel_rows = [], []
    for appr in APPROACHES:
        t0 = time.time()
        Xtr_raw, ytr = X[tr], y[tr]
        info = {"n_train_removed": 0, "geo_evals": None, "k": None}
        if appr == "duplicate":
            keep = duplicate_correction(Xtr_raw, ytr)
            info["n_train_removed"] = int((~keep).sum())
            Xtr_raw, ytr = Xtr_raw[keep], ytr[keep]
        prep = Prep().fit(Xtr_raw)
        Xtr, Xte = prep.transform(Xtr_raw), prep.transform(X[te])
        cols = np.arange(X.shape[1])
        if appr == "geo":
            cols, info["geo_evals"] = geo_select(Xtr, ytr, seed)
        elif appr == "infotheory":
            cols, _ = ig_select(Xtr, ytr, seed)
            info["k"] = int(len(cols))
        sel_time = time.time() - t0
        Xs, ys = smote(Xtr[:, cols], ytr, seed)
        sel_rows.append(dict(dataset=ds, repeat=rep, fold=fold, approach=appr,
                             n_selected=int(len(cols)),
                             selected=";".join(names[c] for c in cols),
                             selection_seconds=round(sel_time, 3), **info))
        for cname, clf in make_classifiers(seed).items():
            t1 = time.time()
            clf.fit(Xs, ys)
            Xt = Xte[:, cols]
            m = metrics(y[te], clf.predict(Xt), scores_for(clf, Xt))
            rows.append(dict(dataset=ds, repeat=rep, fold=fold, approach=appr,
                             classifier=cname, n_train=int(len(ytr)),
                             n_test=int(len(te)), n_selected=int(len(cols)),
                             fit_seconds=round(time.time() - t1, 3), **m))
    return rows, sel_rows


# -------------------------------------------------------------- stats -------
def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return float(gt + 0.5 * eq)


def magnitude(a):
    a_ = max(a, 1 - a)  # symmetric thresholds (Vargha & Delaney 2000)
    return ("negligible" if a_ < 0.56 else "small" if a_ < 0.64
            else "medium" if a_ < 0.71 else "large")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m, run = len(p), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def statistics(df, metric):
    out_f, out_p = [], []
    for (ds, clf), g in df.groupby(["dataset", "classifier"]):
        w = g.pivot_table(index=["repeat", "fold"], columns="approach", values=metric)
        w = w[APPROACHES]
        try:
            fs = stats.friedmanchisquare(*[w[a].values for a in APPROACHES])
            fstat, fp = float(fs.statistic), float(fs.pvalue)
        except ValueError:
            fstat, fp = np.nan, np.nan
        out_f.append(dict(dataset=ds, classifier=clf, metric=metric,
                          friedman_chi2=fstat, friedman_p=fp, n_folds=len(w)))
        pairs = list(combinations(APPROACHES, 2))
        res = []
        for a, b in pairs:
            diff = w[b] - w[a]
            try:
                p = float(stats.wilcoxon(w[b], w[a], zero_method="zsplit").pvalue) \
                    if np.any(diff != 0) else 1.0
            except ValueError:
                p = 1.0
            A = a12(w[b], w[a])
            res.append(dict(dataset=ds, classifier=clf, metric=metric,
                            treatment=b, reference=a,
                            mean_treatment=float(w[b].mean()),
                            mean_reference=float(w[a].mean()),
                            mean_diff=float(diff.mean()),
                            sd_diff=float(diff.std(ddof=1)),
                            A12=A, magnitude=magnitude(A), p_raw=p))
        adj = holm([r["p_raw"] for r in res])
        for r, pa in zip(res, adj):
            r["p_holm"] = float(pa)
        out_p.extend(res)
    return pd.DataFrame(out_f), pd.DataFrame(out_p)


# --------------------------------------------------------------- main -------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    data, logs, tasks = {}, {}, []
    for ds in args.datasets:
        path = os.path.join(args.data, ds.lower() + ".arff")
        Xdf, y = load_arff(path)
        X, yv, names, log = common_cleaning(Xdf, y)
        log["file"] = os.path.basename(path)
        log["sha256"] = sha256(path)
        logs[ds] = log
        data[ds] = (X, yv, names)
        rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                       random_state=SEED)
        for k, (tr, te) in enumerate(rskf.split(X, yv)):
            tasks.append((ds, k // args.folds, k % args.folds, tr, te))
    json.dump(logs, open(os.path.join(args.out, "cleaning_log.json"), "w"), indent=2)

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_fold)(ds, *data[ds], rep, fold, tr, te)
        for ds, rep, fold, tr, te in tasks)
    fold_df = pd.DataFrame([r for rows, _ in results for r in rows])
    sel_df = pd.DataFrame([r for _, rows in results for r in rows])
    fold_df.to_csv(os.path.join(args.out, "fold_level_results.csv"), index=False)
    sel_df.to_csv(os.path.join(args.out, "fold_level_selection.csv"), index=False)

    metric_cols = ["mcc", "auc", "recall", "precision", "f1", "accuracy"]
    summ = fold_df.groupby(["dataset", "approach", "classifier"])[metric_cols] \
        .agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ.reset_index().to_csv(os.path.join(args.out, "table_mean_sd.csv"), index=False)

    fr_all, pw_all = [], []
    for m in ["mcc", "auc", "recall", "f1"]:
        f, p = statistics(fold_df, m)
        fr_all.append(f); pw_all.append(p)
    pd.concat(fr_all).to_csv(os.path.join(args.out, "friedman_tests.csv"), index=False)
    pd.concat(pw_all).to_csv(os.path.join(args.out, "pairwise_tests.csv"), index=False)

    import sklearn, scipy, imblearn
    manifest = dict(
        script=os.path.basename(__file__), script_sha256=sha256(__file__),
        started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        wall_seconds=round(time.time() - t_start, 1),
        cv=dict(type="RepeatedStratifiedKFold", folds=args.folds,
                repeats=args.repeats, seed=SEED),
        fold_seed_rule="SEED + 1000*repeat + fold (SMOTE, classifiers, GEO, inner CV)",
        approaches=APPROACHES, geo=GEO, information_gain=IG, smote_k=SMOTE_K,
        classifiers={k: str(v) for k, v in make_classifiers(SEED).items()},
        datasets=logs,
        environment=dict(python=sys.version.split()[0], platform=platform.platform(),
                         processor=platform.processor() or platform.machine(),
                         cpu_count=os.cpu_count(), numpy=np.__version__,
                         pandas=pd.__version__, scipy=scipy.__version__,
                         sklearn=sklearn.__version__, imblearn=imblearn.__version__,
                         xgboost=(__import__("xgboost").__version__ if HAVE_XGB else None)),
    )
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print("done in", manifest["wall_seconds"], "s")


if __name__ == "__main__":
    main()
