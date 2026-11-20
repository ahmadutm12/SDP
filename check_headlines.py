#!/usr/bin/env python3
"""Print the headline numbers of revision 1 from the tables, for comparison with README.md.

  python check_headlines.py --tables results_rev1/tables
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--tables", default="results_rev1/tables"); a = ap.parse_args()
T = a.tables
e = pd.read_csv(f"{T}/main_effects_vs_none.csv"); m = e[e.metric == "mcc"]; u = e[e.metric == "auc"]
print(f"H1  MCC cells: {len(m)}; CI above 0: {(m.ci_lo > 0).sum()}; CI below 0: {(m.ci_hi < 0).sum()}")
for _, r in m[m.ci_hi < 0].sort_values(["approach", "dataset"]).iterrows():
    print(f"    below 0: {r.approach}/{r.dataset}/{r.classifier} {r.delta:+.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")
print(f"H2  AUC cells: CI above 0: {(u.ci_lo > 0).sum()}; CI below 0: {(u.ci_hi < 0).sum()}")
print("H3  threshold rule (A12 >= 0.64 and dMCC >= 0.05): increases",
      int(((m.A12 >= .64) & (m.delta >= .05)).sum()), "decreases", int(((m.A12 <= .36) & (m.delta <= -.05)).sum()))
print("H4  dMCC range by approach:", {k: (round(float(g.delta.min()), 3), round(float(g.delta.max()), 3)) for k, g in m.groupby("approach")})
s = pd.read_csv(f"{T}/main_mean_sd.csv")
print("H5  best mean MCC per dataset:", {k: round(float(v), 3) for k, v in s.groupby("dataset").mcc_mean.max().items()})
st = pd.read_csv(f"{T}/stability.csv")
print("H6  Nogueira stability:", {f"{r.approach}/{r.dataset}": round(float(r.stability_phi), 3) for _, r in st.iterrows()})
f = pd.read_csv(f"{T}/fix_effect_ig.csv"); f = f[f.metric == "mcc"]
print(f"H7  nesting-fix effect on IG dMCC: {f.fix_effect.min():+.3f} to {f.fix_effect.max():+.3f} (mean {f.fix_effect.mean():+.3f})")
ss = pd.read_csv(f"{T}/selector_sensitivity_summary.csv")
print("H8  selector settings, CI above 0 per setting:", dict(zip(ss.config, ss.ci_above0)))
mt = pd.read_csv(f"{T}/matched_vs_dt.csv")
print(f"H9  matched selection: cells {len(mt)}; CI above 0 {(mt.ci_lo_matched > 0).sum()}; CI below 0 {(mt.ci_hi_matched < 0).sum()}")
nd = pd.read_csv(f"{T}/nodedup_effects.csv"); nd = nd[nd.metric == "mcc"]
print(f"H10 duplicates kept: cells {len(nd)}; CI above 0 {(nd.ci_lo > 0).sum()}; CI below 0 {(nd.ci_hi < 0).sum()}")
rq = pd.read_csv(f"{T}/rq2.csv")
print("H11 RQ2 between-cell variance shares:\n", rq.round(3).to_string(index=False))
