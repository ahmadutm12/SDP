#!/usr/bin/env python3
"""Draw the revision-1 figures from the tables written by make_tables_rev1.py.

  python make_figures_rev1.py --tables results_rev1/tables --out results_rev1/figures

Fig_forest_dMCC.png               difference in MCC from no intervention, corrected 95% CI (main run)
FigS1_cleaning_baseline.png       no-intervention MCC with and without exact-duplicate removal
FigS2_sensitivity_matched.png     selector settings (a) and classifier-matched selection (b)
FigS3_selection_frequency.png     how often each metric was selected, with Nogueira stability
All figures are 16 cm wide, Arial (falls back to DejaVu Sans), text 8-9 pt, 300 dpi.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CM = 1 / 2.54
COL = {"inconsistent": "#4C72B0", "geo": "#DD8452", "infotheory": "#55A868"}
LAB = {"inconsistent": "Inconsistent-record removal", "geo": "GEO feature selection",
       "infotheory": "Information-gain selection"}
CLF = ["DT", "RF", "XGBoost", "SVM"]; APP = ["inconsistent", "geo", "infotheory"]; DS = ["CM1", "JM1", "KC1", "PC1"]


def style():
    mpl.rcParams.update({"savefig.bbox": None, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42})


def forest(T, out):
    e = pd.read_csv(f"{T}/main_effects_vs_none.csv"); e = e[e.metric == "mcc"]
    ypos, y = {}, 0
    for c in CLF:
        for a in APP: ypos[(c, a)] = y; y += 1
        y += 0.8
    fig, axs = plt.subplots(1, 4, figsize=(16 * CM, 9.5 * CM), sharey=True, sharex=True)
    for ax, ds in zip(axs, DS):
        ax.axvspan(-0.05, 0.05, color="0.92", zorder=0); ax.axvline(0, color="0.4", lw=0.8, zorder=1)
        for _, r in e[e.dataset == ds].iterrows():
            yy = ypos[(r.classifier, r.approach)]; sig = (r.ci_lo > 0) | (r.ci_hi < 0)
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=COL[r.approach], lw=1.2, zorder=2)
            ax.plot(r.delta, yy, "o", ms=4.2, mfc=COL[r.approach] if sig else "white", mec=COL[r.approach], mew=1.2, zorder=3)
        ax.set_title(ds, loc="left"); ax.set_xlim(-0.26, 0.26); ax.set_xticks([-0.2, 0, 0.2]); ax.tick_params(axis="y", length=0)
    axs[0].invert_yaxis()
    axs[0].set_yticks([np.mean([ypos[(c, a)] for a in APP]) for c in CLF]); axs[0].set_yticklabels(CLF)
    fig.supxlabel("Difference in MCC from no intervention (corrected 95% CI)", fontsize=9, y=0.155)
    h = [Line2D([], [], color=COL[a], marker="o", lw=1.2, ms=4.2, label=LAB[a]) for a in APP]
    h += [Line2D([], [], color="0.3", marker="o", mfc="white", mec="0.3", lw=0, ms=4.2, label="95% CI includes 0 (open)"),
          Patch(color="0.92", label="\u00b10.05 MCC margin")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, -0.01), columnspacing=1.2, handlelength=1.6)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.25, wspace=0.08)
    fig.savefig(f"{out}/Fig_forest_dMCC.png"); plt.close(fig)


def cleaning(T, out):
    ba = pd.read_csv(f"{T}/cleaning_before_after.csv")
    ovp = f"{T}/train_test_vector_overlap.csv"
    ov = pd.read_csv(ovp).set_index("dataset") if os.path.exists(ovp) else None
    fig, axs = plt.subplots(1, 4, figsize=(16 * CM, 7 * CM), sharey=True)
    for ax, ds in zip(axs, DS):
        b = ba[ba.dataset == ds].set_index("classifier").loc[CLF]; yy = np.arange(4)
        for i, (o, d) in enumerate(zip(b.mcc_original, b.mcc_dedup)): ax.plot([d, o], [i, i], color="0.75", lw=1.5, zorder=1)
        ax.plot(b.mcc_original, yy, "o", color="#8172B3", ms=5, label="Original files (duplicates kept)", zorder=2)
        ax.plot(b.mcc_dedup, yy, "s", color="0.25", ms=4.5, label="Exact duplicates removed", zorder=2)
        t = ds if ov is None else f"{ds}\n{ov.loc[ds, 'False']:.1f}% of test rows have\nan identical training row"
        ax.set_title(t, loc="left", fontsize=8); ax.set_xlim(0, 0.45); ax.set_xticks([0, 0.2, 0.4]); ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(yy); axs[0].set_yticklabels(CLF); axs[0].invert_yaxis()
    fig.supxlabel("Mean MCC with no intervention (50 folds)", fontsize=9, y=0.12)
    hh, ll = axs[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.3, wspace=0.12)
    fig.savefig(f"{out}/FigS1_cleaning_baseline.png"); plt.close(fig)


def sensitivity(T, out):
    order = [("ig_main_bins10_kselected", "IG: 10 bins, k tuned (main)"), ("ig_bins5", "IG: 5 bins"),
             ("ig_bins20", "IG: 20 bins"), ("ig_k5", "IG: k fixed at 5"), ("ig_k10", "IG: k fixed at 10"),
             ("ig_k15", "IG: k fixed at 15"), ("geo_main_p10_i20", "GEO: 10 agents, 20 iter. (main)"),
             ("geo_p20_i50", "GEO: 20 agents, 50 iter.")]
    sv = pd.read_csv(f"{T}/selector_sensitivity_effects.csv"); sv = sv[sv.metric == "mcc"]
    mt = pd.read_csv(f"{T}/matched_vs_dt.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16 * CM, 8.6 * CM), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a1.axvspan(-0.05, 0.05, color="0.92", zorder=0); a1.axvline(0, color="0.4", lw=0.8)
    rng = np.random.default_rng(1)
    for i, (c, l) in enumerate(order):
        d = sv[sv.config == c]; col = COL["geo"] if "geo" in c else COL["infotheory"]
        sig = (d.ci_hi < 0).values; jit = rng.uniform(-0.18, 0.18, len(d))
        a1.scatter(d.delta[~sig], i + jit[~sig], s=14, facecolor="white", edgecolor=col, lw=1, zorder=2)
        a1.scatter(d.delta[sig], i + jit[sig], s=14, facecolor=col, edgecolor=col, lw=1, zorder=2)
        a1.plot([d.delta.median()] * 2, [i - 0.3, i + 0.3], color="0.15", lw=1.5, zorder=3)
    a1.set_yticks(range(len(order))); a1.set_yticklabels([l for _, l in order]); a1.invert_yaxis(); a1.tick_params(axis="y", length=0)
    a1.set_xlabel("\u0394MCC vs no intervention"); a1.set_xlim(-0.16, 0.11); a1.set_xticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
    a1.set_title("No setting yields a CI above zero", loc="left")
    hk = [Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=4, label="CI includes 0"),
          Line2D([], [], marker="o", ls="", color="0.3", ms=4, label="CI below 0"),
          Line2D([], [], color="0.15", lw=1.5, label="Median of 12 cells")]
    a1.legend(handles=hk, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.36, -0.13), handletextpad=0.3, columnspacing=0.9)
    mcol = {"matched_ig_RF": COL["infotheory"], "matched_ig_XGB": COL["infotheory"], "matched_geo_XGB": COL["geo"]}
    mlab = {"matched_ig_RF": "IG\nRF", "matched_ig_XGB": "IG\nXGBoost", "matched_geo_XGB": "GEO\nXGBoost"}
    a2.axhspan(-0.05, 0.05, color="0.92", zorder=0); a2.axhline(0, color="0.4", lw=0.8)
    for j, (cfg, g) in enumerate(mt.groupby("config", sort=False)):
        g = g.set_index("dataset").loc[DS]
        for k, (_, r) in enumerate(g.iterrows()):
            x0 = j * 3 + k * 0.55 - 0.8
            a2.plot([x0, x0 + 0.4], [r.delta_DTtuned, r.delta_matched], color=mcol[cfg], lw=1, zorder=2)
            a2.plot(x0, r.delta_DTtuned, "o", ms=3.5, mfc="white", mec=mcol[cfg], zorder=3)
            a2.plot(x0 + 0.4, r.delta_matched, "o", ms=3.5, color=mcol[cfg], zorder=3)
    a2.set_xticks([j * 3 for j in range(3)]); a2.set_xticklabels([mlab[c] for c in mt.config.unique()]); a2.tick_params(axis="x", length=0)
    a2.set_ylim(-0.13, 0.08); a2.set_yticks([-0.1, -0.05, 0, 0.05]); a2.set_ylabel("\u0394MCC vs no intervention", labelpad=2)
    a2.set_title("Selector matched to classifier", loc="left")
    a2.plot([], [], "o", mfc="white", mec="0.3", ms=3.5, ls="", label="Inner DT objective")
    a2.plot([], [], "o", color="0.3", ms=3.5, ls="", label="Matching classifier")
    a2.legend(loc="lower left", frameon=False, bbox_to_anchor=(-0.02, -0.03), handletextpad=0.3)
    for ax, L, dx in [(a1, "a", -0.62), (a2, "b", -0.28)]:
        ax.text(dx, 1.02, L, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
    fig.subplots_adjust(left=0.25, right=0.985, top=0.91, bottom=0.25, wspace=0.40)
    fig.savefig(f"{out}/FigS2_sensitivity_matched.png"); plt.close(fig)


def frequency(T, out):
    fq = pd.read_csv(f"{T}/selection_frequency.csv"); st = pd.read_csv(f"{T}/stability.csv")
    fq["col"] = fq.approach.map({"infotheory": "IG", "geo": "GEO"}) + " " + fq.dataset
    cols = [f"IG {d}" for d in DS] + [f"GEO {d}" for d in DS]
    H = fq.pivot_table(index="metric", columns="col", values="selection_frequency")[cols]
    H = H.loc[H[cols[:4]].mean(1).sort_values(ascending=False).index]
    phi = st.set_index(["approach", "dataset"]).stability_phi
    fig, ax = plt.subplots(figsize=(16 * CM, 13 * CM))
    im = ax.imshow(H.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.values[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else "black")
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{c}\n\u03a6 = {phi[('infotheory' if c.startswith('IG') else 'geo', c.split()[1])]:.2f}" for c in cols])
    ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index); ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Share of the 50 outer folds in which the metric was selected")
    fig.subplots_adjust(left=0.2, right=0.92, top=0.88, bottom=0.02)
    fig.savefig(f"{out}/FigS3_selection_frequency.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tables", default="results_rev1/tables")
    ap.add_argument("--out", default="results_rev1/figures"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); style()
    for f in (forest, cleaning, sensitivity, frequency): f(a.tables, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
