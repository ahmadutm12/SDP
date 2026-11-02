#!/usr/bin/env python3
"""Manuscript tables from sdp_experiment.py output.
Usage: python make_tables.py RESULTS_DIR  -> writes tables.md and table10.csv"""
import json, os, sys
import pandas as pd

out = sys.argv[1]
DS = ["CM1", "JM1", "KC1", "PC1"]
AP = ["none", "duplicate", "geo", "infotheory"]
LAB = {"none": "No correction", "duplicate": "Duplicate-bug-report",
       "geo": "GEO", "infotheory": "Information-theory"}
CLF = ["DT", "RF", "XGBoost", "SVM"]
ms = pd.read_csv(os.path.join(out, "table_mean_sd.csv")).set_index(
    ["dataset", "approach", "classifier"])
pw = pd.read_csv(os.path.join(out, "pairwise_tests.csv"))
fr = pd.read_csv(os.path.join(out, "friedman_tests.csv"))
sel = pd.read_csv(os.path.join(out, "fold_level_selection.csv"))
log = json.load(open(os.path.join(out, "cleaning_log.json")))

lines = []
def md(title, header, rows):
    lines.append(f"\n### {title}\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")

for metric, name in [("mcc", "Table 6 MCC"), ("auc", "Table 7 ROC-AUC"),
                     ("recall", "Table 8 Recall"), ("f1", "Table 9 F1"),
                     ("precision", "Table S1a Precision"), ("accuracy", "Table S1b Accuracy")]:
    rows = []
    for d in DS:
        for a in AP:
            rows.append([d, LAB[a]] + [
                f"{ms.loc[(d, a, c), metric + '_mean']:.3f} ± {ms.loc[(d, a, c), metric + '_std']:.3f}"
                for c in CLF])
    md(name, ["Dataset", "Approach"] + CLF, rows)

# Table 10: each approach vs none, MCC
t10 = pw[(pw.metric == "mcc") & (pw.reference == "none")]
rows, csv = [], []
for d in DS:
    for a in ["duplicate", "geo", "infotheory"]:
        cells = []
        for c in CLF:
            r = t10[(t10.dataset == d) & (t10.classifier == c) & (t10.treatment == a)].iloc[0]
            star = "*" if r.p_holm < 0.05 else ""
            cells.append(f"{r.mean_diff:+.3f} ({r.A12:.2f}){star}")
            csv.append(dict(dataset=d, approach=a, classifier=c, mean_diff=r.mean_diff,
                            A12=r.A12, magnitude=r.magnitude, p_holm=r.p_holm))
        rows.append([d, LAB[a] + " vs. no correction"] + cells)
md("Table 10 dMCC (A12) vs none; * Holm p<0.05", ["Dataset", "Comparison"] + CLF, rows)
pd.DataFrame(csv).to_csv(os.path.join(out, "table10.csv"), index=False)

# Friedman
rows = [[r.dataset, r.classifier, f"{r.friedman_chi2:.2f}", f"{r.friedman_p:.3g}"]
        for r in fr[fr.metric == "mcc"].itertuples()]
md("Friedman (MCC)", ["Dataset", "Classifier", "chi2", "p"], rows)

# selection summary
g = sel.groupby(["dataset", "approach"]).agg(
    n_selected_mean=("n_selected", "mean"), n_selected_min=("n_selected", "min"),
    n_selected_max=("n_selected", "max"), removed_mean=("n_train_removed", "mean"),
    removed_min=("n_train_removed", "min"), removed_max=("n_train_removed", "max"),
    sel_sec_mean=("selection_seconds", "mean"), geo_evals=("geo_evals", "mean"))
g.round(3).to_csv(os.path.join(out, "selection_summary.csv"))
lines.append("\n### Selection summary\n")
lines.append(g.round(3).to_markdown())
# most frequently selected metrics
for a in ["geo", "infotheory"]:
    lines.append(f"\n### Most frequently selected metrics: {a}\n")
    for d in DS:
        s = sel[(sel.dataset == d) & (sel.approach == a)]["selected"].str.split(";").explode()
        top = (s.value_counts() / len(sel[(sel.dataset == d) & (sel.approach == a)])).head(6)
        lines.append(f"- {d}: " + ", ".join(f"{k} ({v:.0%})" for k, v in top.items()))
lines.append("\n### Cleaning log\n```\n" + json.dumps(log, indent=1) + "\n```")
open(os.path.join(out, "tables.md"), "w").write("\n".join(lines))
print("\n".join(lines))
