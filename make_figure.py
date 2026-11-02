#!/usr/bin/env python3
"""Fig. 2: MCC and ROC-AUC (mean ± SD over all folds) by dataset, approach and
classifier, drawn from fold_level_results.csv produced by sdp_experiment.py.
Usage: python make_figure.py RESULTS_DIR"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
df = pd.read_csv(os.path.join(out, "fold_level_results.csv"))
DS = ["CM1", "JM1", "KC1", "PC1"]
AP = ["none", "duplicate", "geo", "infotheory"]
LAB = {"none": "No correction", "duplicate": "Duplicate-bug-report",
       "geo": "GEO", "infotheory": "Information-theory"}
COL = {"none": "#8a8a85", "duplicate": "#2a78d6", "geo": "#eb6834",
       "infotheory": "#1baf7a"}
CLF = [c for c in ["DT", "RF", "XGBoost", "SVM"] if c in set(df.classifier)]
MET = [("mcc", "MCC", 0.0), ("auc", "ROC-AUC", 0.5)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555", "axes.linewidth": 0.6,
                     "xtick.color": "#333", "ytick.color": "#333"})
fig, axes = plt.subplots(len(MET), len(CLF), figsize=(7.2, 4.6), sharey="row")
g = df.groupby(["metric_dummy"] if False else ["classifier", "dataset", "approach"])
w = 0.19
x = np.arange(len(DS))
for r, (m, mlab, ref) in enumerate(MET):
    s = df.groupby(["classifier", "dataset", "approach"])[m].agg(["mean", "std"])
    for c, clf in enumerate(CLF):
        ax = axes[r, c]
        for k, a in enumerate(AP):
            mu = [s.loc[(clf, d, a), "mean"] for d in DS]
            sd = [s.loc[(clf, d, a), "std"] for d in DS]
            ax.bar(x + (k - 1.5) * w, mu, w * 0.9, yerr=sd, color=COL[a],
                   label=LAB[a], error_kw=dict(lw=0.6, capsize=1.2, ecolor="#333"))
        ax.axhline(ref, color="#333", lw=0.7, ls="--")
        ax.set_xticks(x, DS)
        ax.grid(axis="y", color="#e5e5e5", lw=0.5)
        ax.set_axisbelow(True)
        if r == 0:
            ax.set_title(clf, fontsize=9)
        if c == 0:
            ax.set_ylabel(mlab)
    if m == "auc":
        axes[r, 0].set_ylim(0.4, 1.0)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"fig2_mcc_auc.{ext}"), dpi=600)
print("saved")
