#!/usr/bin/env python3
"""
Revision 1 (CMC 96252) of the leakage-controlled comparison of data-quality
interventions for software defect prediction on the PROMISE versions of NASA
CM1, JM1, KC1 and PC1.

Changes relative to sdp_experiment.py (v03 submission):
  1. Properly nested inner selection (Reviewer 3). Inside every inner training
     split, median imputation, standardisation, equal-frequency discretisation and
     the information-gain ranking are fitted on that split only and evaluated on
     the inner validation split. The GEO fitness uses the same per-split
     preprocessing. After k (IG) or the mask (GEO) is chosen, the final ranking /
     mask is applied to the outer training fold, preprocessed on that fold only.
     Outer test folds never enter any selection step (as before).
  2. The inner objective classifier is configurable (DT as in v03; RF or XGBoost
     with 100 trees for the classifier-matched sensitivity analysis).
  3. Options for the sensitivity analyses: common exact-duplicate removal on/off,
     IG bins, fixed k, GEO population and iterations.
  4. Training-fold MCC (on the non-oversampled training fold) is recorded for the
     overfitting check.
Condition names: none, inconsistent (inconsistent-record removal, formerly
"duplicate-bug-report-based"), geo, infotheory.
"""
import hashlib, io, re, time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATASETS = ["CM1", "JM1", "KC1", "PC1"]
APPROACHES = ["none", "inconsistent", "geo", "infotheory"]
SEED = 42
SMOTE_K = 5


@dataclass
class Config:
    dedup: bool = True                 # common exact-duplicate removal before splitting
    inner_clf: str = "DT"              # inner objective for GEO fitness and IG k choice
    geo_pop: int = 10
    geo_iters: int = 20
    geo_alpha: float = 0.01
    geo_bound: float = 6.0
    pa: tuple = (0.5, 2.0)
    pc: tuple = (1.0, 0.5)
    inner_folds: int = 3
    ig_bins: int = 10
    k_grid: tuple = (5, 10, 15)
    nested: bool = True                # False reproduces the v03 (non-nested) IG selection
    fixed_k: int | None = None         # if set, IG keeps top fixed_k (no inner choice)
    approaches: tuple = tuple(APPROACHES)
    classifiers: tuple = ("DT", "RF", "XGBoost", "SVM")   # outer classifiers evaluated
    label: str = "main"

    def asdict(self):
        return asdict(self)


# ---------------------------------------------------------------- data ------
def load_arff(path):
    lines = open(path, errors="replace").read().splitlines()
    attrs = [re.split(r"\s+", l.strip())[1] for l in lines if l.lower().startswith("@attribute")]
    start = [i for i, l in enumerate(lines) if l.lower().startswith("@data")][0]
    rows = [l for l in lines[start + 1:] if l.strip() and not l.startswith("%")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=attrs, na_values=["?"])
    y = df.iloc[:, -1].astype(str).str.strip().str.lower().map({"true": 1, "yes": 1, "false": 0, "no": 0})
    return df.iloc[:, :-1].astype(float), y


def common_cleaning(X, y, dedup=True):
    log = {"modules_raw": int(len(y)), "defective_raw": int(y.sum())}
    keep = y.notna()
    X, y = X[keep], y[keep].astype(int)
    log["removed_missing_label"] = int((~keep).sum())
    df = X.copy(); df["__y"] = y.values
    dup = df.duplicated(keep="first")
    log["exact_duplicates_present"] = int(dup.sum())
    if dedup:
        df = df[~dup].reset_index(drop=True)
    log["removed_exact_duplicates"] = int(dup.sum()) if dedup else 0
    X, y = df.drop(columns="__y"), df["__y"].astype(int)
    log.update(modules_clean=int(len(y)), defective_clean=int(y.sum()), defect_rate_clean=float(y.mean()),
               missing_metric_cells=int(X.isna().sum().sum()), n_metrics=int(X.shape[1]))
    g = df.groupby(list(X.columns), dropna=False)["__y"].transform("nunique")
    log["inconsistent_records"] = int((g > 1).sum())
    return X.values, y.values, list(X.columns), log


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------- preprocessing ------
class Prep:
    """Median imputation + standardisation, fitted on the data passed to fit()."""
    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        self.sc = StandardScaler().fit(np.where(np.isnan(X), self.med, X))
        return self

    def transform(self, X):
        return self.sc.transform(np.where(np.isnan(X), self.med, X))


def smote(X, y, seed):
    k = min(SMOTE_K, int(y.sum()) - 1)
    if k < 1:
        return X, y
    return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)


def make_inner(name, seed):
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=100, objective="binary:logistic", eval_metric="logloss",
                             random_state=seed, n_jobs=1, tree_method="hist")
    if name == "SVM":
        return SVC(kernel="rbf", random_state=seed)
    raise ValueError(name)


def make_classifiers(seed):
    return {
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "SVM": SVC(kernel="rbf", random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, objective="binary:logistic", eval_metric="logloss",
                                 random_state=seed, n_jobs=1, tree_method="hist"),
    }


def inner_splits(X_raw, y, seed, n_folds):
    """Inner splits of the outer training fold; preprocessing fitted per inner split."""
    out = []
    for tr, va in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_raw, y):
        p = Prep().fit(X_raw[tr])
        out.append((p.transform(X_raw[tr]), y[tr], p.transform(X_raw[va]), y[va]))
    return out


def inner_score(splits, cols, seed, inner_clf):
    s = []
    for Xtr, ytr, Xva, yva in splits:
        Xs, ys = smote(Xtr[:, cols], ytr, seed)
        clf = make_inner(inner_clf, seed).fit(Xs, ys)
        s.append(matthews_corrcoef(yva, clf.predict(Xva[:, cols])))
    return float(np.mean(s))


# --------------------------------------------------------- approaches -------
def inconsistent_keep(X_raw, y):
    """Keep mask: drop every training record whose full metric vector also occurs with
    the opposite label (label-conflict filtering)."""
    key = pd.util.hash_pandas_object(pd.DataFrame(X_raw).fillna(-1e300), index=False).values
    n_lab = pd.DataFrame({"k": key, "y": y}).groupby("k")["y"].transform("nunique").values
    return n_lab == 1


def entropy(labels):
    p = np.bincount(labels, minlength=2) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def info_gain(X, y, bins):
    H = entropy(y); out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
        v = np.searchsorted(edges, X[:, j], side="right")
        out[j] = H - sum((v == u).mean() * entropy(y[v == u]) for u in np.unique(v))
    return out


def ig_select(X_raw_tr, y_tr, X_prep_tr, seed, cfg):
    """Returns selected column indices, chosen k, inner score per k."""
    d = X_raw_tr.shape[1]
    final_order = np.argsort(-info_gain(X_prep_tr, y_tr, cfg.ig_bins), kind="stable")
    if cfg.fixed_k is not None:
        k = min(cfg.fixed_k, d)
        return np.sort(final_order[:k]), k, {}
    if cfg.nested:
        splits = inner_splits(X_raw_tr, y_tr, seed, cfg.inner_folds)
    else:  # v03 behaviour: outer-fitted preprocessing and one ranking from the whole outer training fold
        splits = [(X_prep_tr[a], y_tr[a], X_prep_tr[b], y_tr[b]) for a, b in
                  StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=seed).split(X_prep_tr, y_tr)]
    scores = {k: [] for k in cfg.k_grid}
    for Xtr, ytr, Xva, yva in splits:
        order = (np.argsort(-info_gain(Xtr, ytr, cfg.ig_bins), kind="stable")  # ranking on inner train only
                 if cfg.nested else final_order)
        for k in cfg.k_grid:
            cols = order[:min(k, d)]
            Xs, ys = smote(Xtr[:, cols], ytr, seed)
            clf = make_inner(cfg.inner_clf, seed).fit(Xs, ys)
            scores[k].append(matthews_corrcoef(yva, clf.predict(Xva[:, cols])))
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    best_k = max(cfg.k_grid, key=lambda k: (mean[k], -k))       # ties -> smaller k
    best_k = min(best_k, d)
    return np.sort(final_order[:best_k]), best_k, mean


def geo_select(X_raw_tr, y_tr, seed, cfg):
    rng = np.random.default_rng(seed)
    d = X_raw_tr.shape[1]; P, T, B = cfg.geo_pop, cfg.geo_iters, cfg.geo_bound
    splits = inner_splits(X_raw_tr, y_tr, seed, cfg.inner_folds)
    cache = {}

    def mask_of(pos):
        m = (1.0 / (1.0 + np.exp(-pos))) > 0.5
        if not m.any():
            m[np.argmax(pos)] = True
        return m

    def fitness(pos):
        m = mask_of(pos); key = m.tobytes()
        if key not in cache:
            cache[key] = inner_score(splits, np.flatnonzero(m), seed, cfg.inner_clf) - cfg.geo_alpha * m.sum() / d
        return cache[key]

    pos = rng.uniform(-B, B, size=(P, d))
    fit = np.array([fitness(p) for p in pos]); mem, memfit = pos.copy(), fit.copy()
    for t in range(1, T + 1):
        pa = cfg.pa[0] + (t / T) * (cfg.pa[1] - cfg.pa[0])
        pc = cfg.pc[0] - (t / T) * (cfg.pc[0] - cfg.pc[1])
        for i in range(P):
            f = rng.integers(P); A = mem[f] - pos[i]; nA = np.linalg.norm(A)
            if nA < 1e-12:
                continue
            r = rng.normal(size=d); C = r - (r @ A) / (nA ** 2) * A; nC = np.linalg.norm(C)
            step = rng.random() * pa * A / nA
            if nC > 1e-12:
                step = step + rng.random() * pc * C / nC
            pos[i] = np.clip(pos[i] + step, -B, B)
            fi = fitness(pos[i])
            if fi > memfit[i]:
                mem[i], memfit[i] = pos[i].copy(), fi
    return np.flatnonzero(mask_of(mem[int(np.argmax(memfit))])), len(cache)


# --------------------------------------------------------------- fold -------
def scores_for(clf, X):
    return clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)


def metrics(y, yhat, score):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn), precision=prec, recall=rec, f1=f1,
                mcc=matthews_corrcoef(y, yhat), auc=roc_auc_score(y, score), accuracy=(tp + tn) / len(y))


def run_fold(ds, X, y, names, rep, fold, tr, te, cfg):
    seed = SEED + 1000 * rep + fold
    rows, sel = [], []
    for appr in cfg.approaches:
        t0 = time.time()
        Xtr_raw, ytr = X[tr], y[tr]
        info = dict(n_train_removed=0, geo_evals=None, k=None, ig_inner=None)
        if appr == "inconsistent":
            keep = inconsistent_keep(Xtr_raw, ytr)
            info["n_train_removed"] = int((~keep).sum()); Xtr_raw, ytr = Xtr_raw[keep], ytr[keep]
        prep = Prep().fit(Xtr_raw)
        Xtr, Xte = prep.transform(Xtr_raw), prep.transform(X[te])
        cols = np.arange(X.shape[1])
        if appr == "geo":
            cols, info["geo_evals"] = geo_select(Xtr_raw, ytr, seed, cfg)
        elif appr == "infotheory":
            cols, info["k"], inner = ig_select(Xtr_raw, ytr, Xtr, seed, cfg)
            info["ig_inner"] = ";".join(f"{k}:{v:.4f}" for k, v in inner.items())
        sel_t = time.time() - t0
        Xs, ys = smote(Xtr[:, cols], ytr, seed)
        sel.append(dict(dataset=ds, repeat=rep, fold=fold, approach=appr, config=cfg.label,
                        n_selected=int(len(cols)), selected=";".join(names[c] for c in cols),
                        selection_seconds=round(sel_t, 3), **info))
        for cname, clf in make_classifiers(seed).items():
            if cname not in cfg.classifiers:
                continue
            clf.fit(Xs, ys)
            m = metrics(y[te], clf.predict(Xte[:, cols]), scores_for(clf, Xte[:, cols]))
            m["train_mcc"] = matthews_corrcoef(ytr, clf.predict(Xtr[:, cols]))
            rows.append(dict(dataset=ds, repeat=rep, fold=fold, approach=appr, classifier=cname,
                             config=cfg.label, n_train=int(len(ytr)), n_test=int(len(te)),
                             n_selected=int(len(cols)), **m))
    return rows, sel


def build_tasks(data_dir, cfg, datasets=DATASETS, repeats=5, folds=10):
    data, logs, tasks = {}, {}, []
    for ds in datasets:
        path = f"{data_dir}/{ds.lower()}.arff"
        Xdf, y = load_arff(path)
        X, yv, names, log = common_cleaning(Xdf, y, dedup=cfg.dedup)
        log["sha256"] = sha256(path); logs[ds] = log; data[ds] = (X, yv, names)
        for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                                             random_state=SEED).split(X, yv)):
            tasks.append((ds, k // folds, k % folds, tr, te))
    return data, logs, tasks
