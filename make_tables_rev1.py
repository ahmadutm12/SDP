#!/usr/bin/env python3
"""Build every revision-1 analysis table from the fold-level outputs.

  python make_tables_rev1.py --results results_rev1 --out results_rev1/tables

Needs folds_/selection_ files for: main, v03ref_ig, nodedup, nodedup_svm_none,
matched_ig_RF, matched_ig_XGB, matched_geo_XGB, ig_bins5, ig_bins20, ig_k5, ig_k10,
ig_k15, geo_p20_i50, plus kc1_permutation_importance.csv (kc1_importance.py) and
train_test_vector_overlap.csv (overlap_check.py) if those tables are wanted.
All confidence intervals are the corrected repeated-CV t intervals in analysis_rev1.py.
"""
import argparse, json, os
import numpy as np, pandas as pd
import analysis_rev1 as A

SENS = [("ig_bins5", "infotheory"), ("ig_bins20", "infotheory"), ("ig_k5", "infotheory"),
        ("ig_k10", "infotheory"), ("ig_k15", "infotheory"), ("geo_p20_i50", "geo")]
MATCHED = [("matched_ig_RF", "infotheory", "RF"), ("matched_ig_XGB", "infotheory", "XGBoost"),
           ("matched_geo_XGB", "geo", "XGBoost")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results_rev1")
    ap.add_argument("--out", default=None); a = ap.parse_args()
    R = a.results; T = a.out or f"{R}/tables"; os.makedirs(T, exist_ok=True)
    rd = lambda n: pd.read_csv(f"{R}/folds_{n}.csv")
    fm = rd("main"); sm = pd.read_csv(f"{R}/selection_main.csv")
    names = sm[sm.approach == "none"].selected.iloc[0].split(";")

    # main analysis
    A.summary_table(fm).to_csv(f"{T}/main_mean_sd.csv", index=False)
    em = A.effects(fm, metrics=("mcc", "auc", "recall", "f1")); em.to_csv(f"{T}/main_effects_vs_none.csv", index=False)
    fr, pw = A.pairwise_holm(fm)
    fr.to_csv(f"{T}/main_friedman_mcc.csv", index=False); pw.to_csv(f"{T}/main_wilcoxon_holm_mcc.csv", index=False)
    pd.concat([A.threshold_grid(em), A.threshold_grid(em, col="A12_paired")]).to_csv(f"{T}/threshold_grid.csv", index=False)
    A.rq2_decomposition(em).merge(A.rq2_mixed(fm), on="approach").to_csv(f"{T}/rq2.csv", index=False)
    A.overfitting(fm).to_csv(f"{T}/overfitting.csv", index=False)
    stm, fqm = A.stability_tables(sm, names)
    stm.to_csv(f"{T}/stability.csv", index=False); fqm.to_csv(f"{T}/selection_frequency.csv", index=False)

    # effect of the nesting fix (same folds, same environment)
    fv = rd("v03ref_ig"); sv = pd.read_csv(f"{R}/selection_v03ref_ig.csv")
    ev = A.effects(fv, ref_fl=fm, metrics=("mcc", "auc"))
    fix = ev.merge(em[em.approach == "infotheory"], on=["dataset", "classifier", "metric", "approach"],
                   suffixes=("_unnested", "_nested"))
    fix["fix_effect"] = fix.delta_nested - fix.delta_unnested
    fix[["dataset", "classifier", "metric", "delta_unnested", "ci_lo_unnested", "ci_hi_unnested",
         "delta_nested", "ci_lo_nested", "ci_hi_nested", "fix_effect"]].to_csv(f"{T}/fix_effect_ig.csv", index=False)
    mk = sv[["dataset", "repeat", "fold", "k"]].merge(
        sm[sm.approach == "infotheory"][["dataset", "repeat", "fold", "k"]],
        on=["dataset", "repeat", "fold"], suffixes=("_unnested", "_nested"))
    mk["changed"] = mk.k_unnested != mk.k_nested
    mk.groupby("dataset").agg(k_unnested=("k_unnested", "mean"), k_nested=("k_nested", "mean"),
                              pct_changed=("changed", lambda s: 100 * s.mean())).reset_index() \
        .to_csv(f"{T}/fix_k_changes.csv", index=False)

    # cleaning sensitivity (exact duplicates kept)
    fn = pd.concat([rd("nodedup"), rd("nodedup_svm_none")])
    ld = json.load(open(f"{R}/run_main.json"))["logs"]; ln = json.load(open(f"{R}/run_nodedup.json"))["logs"]
    rows = []
    for ds in A.pd.unique(fm.dataset):
        for clf in ["DT", "RF", "XGBoost", "SVM"]:
            d = fm[(fm.dataset == ds) & (fm.classifier == clf) & (fm.approach == "none")]
            o = fn[(fn.dataset == ds) & (fn.classifier == clf) & (fn.approach == "none")]
            rows.append(dict(dataset=ds, classifier=clf, modules_original=ln[ds]["modules_clean"],
                             defect_rate_original=ln[ds]["defect_rate_clean"], modules_dedup=ld[ds]["modules_clean"],
                             defect_rate_dedup=ld[ds]["defect_rate_clean"], exact_duplicates=ln[ds]["exact_duplicates_present"],
                             mcc_original=o.mcc.mean(), mcc_original_sd=o.mcc.std(), mcc_dedup=d.mcc.mean(),
                             mcc_dedup_sd=d.mcc.std(), auc_original=o.auc.mean(), auc_dedup=d.auc.mean()))
    pd.DataFrame(rows).to_csv(f"{T}/cleaning_before_after.csv", index=False)
    A.effects(rd("nodedup"), metrics=("mcc", "auc")).to_csv(f"{T}/nodedup_effects.csv", index=False)

    # selector settings (reference: main 'none'; DT, RF, XGBoost)
    parts = []
    for cfg, appr in SENS:
        e = A.effects(rd(cfg), ref_fl=fm, metrics=("mcc", "auc")); e["config"] = cfg; parts.append(e)
    base = em[em.approach.isin(["infotheory", "geo"]) & em.classifier.isin(["DT", "RF", "XGBoost"])
              & em.metric.isin(["mcc", "auc"])].copy()
    base["config"] = np.where(base.approach == "geo", "geo_main_p10_i20", "ig_main_bins10_kselected")
    sens = pd.concat(parts + [base]); sens.to_csv(f"{T}/selector_sensitivity_effects.csv", index=False)
    s = sens[sens.metric == "mcc"]; g = s.groupby("config")
    summ = g.agg(n=("delta", "size"), mean_delta=("delta", "mean"), min_delta=("delta", "min"),
                 max_delta=("delta", "max"), ci_above0=("ci_lo", lambda v: (v > 0).sum()),
                 ci_below0=("ci_hi", lambda v: (v < 0).sum()))
    summ["rule_increase"] = g.apply(lambda x: ((x.A12 >= .64) & (x.delta >= .05)).sum())
    summ["rule_decrease"] = g.apply(lambda x: ((x.A12 <= .36) & (x.delta <= -.05)).sum())
    summ.to_csv(f"{T}/selector_sensitivity_summary.csv")
    st = []
    for cfg in ["ig_bins5", "ig_bins20", "geo_p20_i50", "matched_ig_RF", "matched_ig_XGB", "matched_geo_XGB", "nodedup"]:
        x, _ = A.stability_tables(pd.read_csv(f"{R}/selection_{cfg}.csv"), names); x["config"] = cfg; st.append(x)
    pd.concat(st + [stm.assign(config="main")]).to_csv(f"{T}/stability_sensitivity.csv", index=False)

    # classifier-matched selection
    mt = []
    for cfg, appr, clf in MATCHED:
        e = A.effects(rd(cfg), ref_fl=fm[fm.classifier == clf], metrics=("mcc", "auc")); e = e[e.metric == "mcc"]
        b = em[(em.approach == appr) & (em.classifier == clf) & (em.metric == "mcc")]
        m = e.merge(b, on=["dataset", "classifier", "metric", "approach"], suffixes=("_matched", "_DTtuned"))
        m["config"] = cfg; mt.append(m)
    pd.concat(mt).to_csv(f"{T}/matched_vs_dt.csv", index=False)

    # KC1: permutation importance (all 21 metrics, XGBoost) against IG drop rate
    p = f"{R}/kc1_permutation_importance.csv"
    if os.path.exists(p):
        imp = pd.read_csv(p)
        pi = imp.groupby(["metric_scored", "feature"]).importance_mean.agg(["mean", "std"]).unstack(0)
        pi.columns = [f"{b}_{a}" for a, b in pi.columns]
        ig = fqm[(fqm.dataset == "KC1") & (fqm.approach == "infotheory")].set_index("metric").selection_frequency
        kt = pi.join(ig.rename("IG_selection_freq")); kt["IG_drop_rate"] = 1 - kt.IG_selection_freq
        kt.sort_values("mcc_mean", ascending=False).to_csv(f"{T}/kc1_xgb_importance_vs_ig_drop.csv")
    print("tables written to", T)


if __name__ == "__main__":
    main()
