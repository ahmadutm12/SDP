#!/usr/bin/env python3
"""
Statistical analysis for Revision 1 (CMC 96252). Reads fold-level CSVs produced by
sdp_experiment_v2.py and writes the tables used in the revised manuscript and the
Supplementary Materials.

Uncertainty: the corrected repeated k-fold CV t-test of Bouckaert & Frank (2004),
which applies the variance correction of Nadeau & Bengio (2003):
    t = mean(d) / sqrt((1/(k*r) + n_test/n_train) * var(d)),  df = k*r - 1,
where d are the paired per-fold differences (treatment - no correction). For
repeated k-fold CV this correction is an approximation (it was derived for random
subsampling), so intervals and p-values are reported as approximate.
Effect size: Vargha-Delaney A12 computed over all fold pairs (unpaired; the
definition used in v03) and, as a sensitivity check, the paired version
P(d > 0) + 0.5 P(d = 0).
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

REF = "none"
METRICS = ["mcc", "auc", "recall", "precision", "f1", "accuracy"]


def a12_unpaired(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())


def a12_paired(d):
    d = np.asarray(d)
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def corrected_ci(d, n_train, n_test, level=0.95):
    d = np.asarray(d, float); J = len(d)
    m, v = d.mean(), d.var(ddof=1)
    se = np.sqrt((1.0 / J + n_test / n_train) * v)
    if se == 0:
        return m, m, m, 1.0 if m == 0 else 0.0
    t = stats.t.ppf(0.5 + level / 2, J - 1)
    p = 2 * stats.t.sf(abs(m / se), J - 1)
    return m, m - t * se, m + t * se, p


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def summary_table(fl):
    g = fl.groupby(["dataset", "approach", "classifier"])[[m for m in METRICS if m in fl]]
    s = g.agg(["mean", "std"]); s.columns = [f"{a}_{b}" for a, b in s.columns]
    return s.reset_index()


def effects(fl, ref_fl=None, metrics=("mcc", "auc"), margin=0.05):
    """Paired effects of each approach vs no correction. ref_fl supplies the
    no-correction folds when fl holds only a treatment (sensitivity runs)."""
    base = fl if ref_fl is None else pd.concat([fl, ref_fl[ref_fl.approach == REF]])
    rows = []
    for (ds, clf), g in base.groupby(["dataset", "classifier"]):
        for m in metrics:
            w = g.pivot_table(index=["repeat", "fold"], columns="approach", values=m)
            ntr = g[g.approach == REF].n_train.mean(); nte = g[g.approach == REF].n_test.mean()
            for tr in [a for a in w.columns if a != REF]:
                d = (w[tr] - w[REF]).dropna()
                mean, lo, hi, p_c = corrected_ci(d.values, ntr, nte)
                pw = stats.wilcoxon(w[tr], w[REF], zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
                rows.append(dict(dataset=ds, classifier=clf, metric=m, approach=tr, n_folds=len(d),
                                 mean_ref=float(w[REF].mean()), mean_treat=float(w[tr].mean()),
                                 delta=mean, ci_lo=lo, ci_hi=hi, p_corrected=p_c, p_wilcoxon=float(pw),
                                 A12=a12_unpaired(w[tr], w[REF]), A12_paired=a12_paired(d.values)))
    e = pd.DataFrame(rows)
    e["ci_excludes_0"] = (e.ci_lo > 0) | (e.ci_hi < 0)
    e["ci_reading"] = np.select(
        [(e.ci_lo > -margin) & (e.ci_hi < margin), e.ci_hi < margin, e.ci_lo > -margin],
        [f"within ±{margin}", f"excludes gain ≥ {margin}", f"excludes loss ≥ {margin}"], "inconclusive")
    return e


def pairwise_holm(fl, metric="mcc"):
    """Descriptive Friedman + Wilcoxon (Holm over all six pairs), as in v03, for the supplement."""
    out_f, out_p = [], []
    apps = sorted(fl.approach.unique(), key=lambda a: (a != REF, a))
    for (ds, clf), g in fl.groupby(["dataset", "classifier"]):
        w = g.pivot_table(index=["repeat", "fold"], columns="approach", values=metric)[apps]
        fs = stats.friedmanchisquare(*[w[a].values for a in apps])
        out_f.append(dict(dataset=ds, classifier=clf, metric=metric, chi2=fs.statistic, p=fs.pvalue))
        res = []
        for a, b in combinations(apps, 2):
            d = w[b] - w[a]
            p = stats.wilcoxon(w[b], w[a], zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
            res.append(dict(dataset=ds, classifier=clf, metric=metric, treatment=b, reference=a,
                            mean_diff=d.mean(), A12=a12_unpaired(w[b], w[a]), p_raw=p))
        for r, pa in zip(res, holm([r["p_raw"] for r in res])):
            r["p_holm"] = pa
        out_p += res
    return pd.DataFrame(out_f), pd.DataFrame(out_p)


def threshold_grid(e, metric="mcc", a_grid=(0.56, 0.64, 0.71), d_grid=(0.02, 0.05, 0.10), col="A12"):
    x = e[e.metric == metric]; rows = []
    for a in a_grid:
        for dd in d_grid:
            inc = ((x[col] >= a) & (x.delta >= dd)); dec = ((x[col] <= 1 - a) & (x.delta <= -dd))
            rows.append(dict(A12_threshold=a, dMCC_threshold=dd, effect_size=col, n_comparisons=len(x),
                             n_increase=int(inc.sum()), n_decrease=int(dec.sum()),
                             increases=";".join(f"{r.approach}/{r.dataset}/{r.classifier}" for r in x[inc].itertuples()),
                             decreases=";".join(f"{r.approach}/{r.dataset}/{r.classifier}" for r in x[dec].itertuples())))
    return pd.DataFrame(rows)


def nogueira_stability(sel, d):
    """Stability estimator of Nogueira, Sechidis & Brown (2018), Eq. (2)."""
    Z = np.asarray(sel, float); M = Z.shape[0]
    p = Z.mean(axis=0); s2 = M / (M - 1) * p * (1 - p); kbar = Z.sum(axis=1).mean()
    den = (kbar / d) * (1 - kbar / d)
    return float(1 - s2.mean() / den) if den > 0 else np.nan


def stability_tables(selection, names):
    rows, freq = [], []
    for (ds, appr), g in selection[selection.approach.isin(["geo", "infotheory"])].groupby(["dataset", "approach"]):
        # metric names are compared case-insensitively (PC1.arff capitalises nine of them, e.g. 'iv(G)', 'N')
        low = [n.lower() for n in names]
        Z = np.array([[n in {x.lower() for x in s.split(";")} for n in low] for s in g.selected])
        rows.append(dict(dataset=ds, approach=appr, n_folds=len(g), mean_selected=Z.sum(1).mean(),
                         stability_phi=nogueira_stability(Z, len(names))))
        for n, f in zip(names, Z.mean(0)):
            freq.append(dict(dataset=ds, approach=appr, metric=n, selection_frequency=f))
    return pd.DataFrame(rows), pd.DataFrame(freq)


def rq2_decomposition(e, metric="mcc"):
    """Two-way decomposition of the 16 cell-mean effects (dataset x classifier) per approach:
    share of between-cell variation due to dataset, classifier and their interaction."""
    rows = []
    for appr, x in e[e.metric == metric].groupby("approach"):
        t = x.pivot_table(index="dataset", columns="classifier", values="delta")
        grand = t.values.mean()
        ss_ds = t.shape[1] * ((t.mean(1) - grand) ** 2).sum()
        ss_cl = t.shape[0] * ((t.mean(0) - grand) ** 2).sum()
        ss_tot = ((t.values - grand) ** 2).sum(); ss_int = ss_tot - ss_ds - ss_cl
        rows.append(dict(approach=appr, ss_total=ss_tot, share_dataset=ss_ds / ss_tot,
                         share_classifier=ss_cl / ss_tot, share_interaction=ss_int / ss_tot,
                         range_dataset_means=float(t.mean(1).max() - t.mean(1).min()),
                         range_classifier_means=float(t.mean(0).max() - t.mean(0).min())))
    return pd.DataFrame(rows)


def rq2_mixed(fl, metric="mcc"):
    """Per-fold paired differences d ~ dataset * classifier, OLS with standard errors
    clustered by outer fold (each fold contributes one d per classifier). Joint Wald tests
    for all classifier terms (main + interaction) and all dataset terms (main + interaction).
    A random-intercept mixed model was attempted first but its fold variance lay on the
    boundary (singular fit), so the cluster-robust model is reported. Descriptive only:
    folds of repeated CV overlap, so the tests are approximate."""
    import statsmodels.formula.api as smf
    rows = []
    for appr in [a for a in fl.approach.unique() if a != REF]:
        recs = []
        for (ds, clf), g in fl.groupby(["dataset", "classifier"]):
            w = g.pivot_table(index=["repeat", "fold"], columns="approach", values=metric)
            for (rep, f), dv in (w[appr] - w[REF]).items():
                recs.append(dict(dataset=ds, classifier=clf, fid=f"{ds}_{rep}_{f}", d=dv))
        D = pd.DataFrame(recs)
        res = smf.ols("d ~ C(dataset) * C(classifier)", D).fit(cov_type="cluster", cov_kwds={"groups": D["fid"]})
        names = res.params.index
        cl_terms = [n for n in names if "classifier" in n]
        ds_terms = [n for n in names if "dataset" in n]
        def wald(terms):
            R = np.zeros((len(terms), len(names)))
            for k, t in enumerate(terms):
                R[k, list(names).index(t)] = 1
            w = res.wald_test(R, scalar=True)
            return float(w.statistic), float(w.pvalue), len(terms)
        f_cl, p_cl, k_cl = wald(cl_terms); f_ds, p_ds, k_ds = wald(ds_terms)
        rows.append(dict(approach=appr, n=len(D), Wald_classifier=f_cl, df_classifier=k_cl, p_classifier=p_cl,
                         Wald_dataset=f_ds, df_dataset=k_ds, p_dataset=p_ds, R2=res.rsquared))
    return pd.DataFrame(rows)


def overfitting(fl):
    if "train_mcc" not in fl:
        return pd.DataFrame()
    fl = fl.assign(gap=fl.train_mcc - fl.mcc)
    s = fl.groupby(["classifier", "approach"]).agg(train_mcc=("train_mcc", "mean"), test_mcc=("mcc", "mean"),
                                                    gap_mean=("gap", "mean"), gap_sd=("gap", "std"))
    return s.reset_index()
