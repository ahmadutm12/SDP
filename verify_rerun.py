#!/usr/bin/env python3
"""Compare a rerun with the reference outputs shipped in results_rev1/.

  python verify_rerun.py --rerun quickcheck --quick          # after run_experiments.py --quick
  python verify_rerun.py --rerun my_results                  # after a full rerun of one or more configs

For every folds_<label>.csv present in --rerun, the script matches rows to the reference
on (dataset, repeat, fold, approach, classifier) and reports the largest absolute
difference in MCC, ROC-AUC, F1 and training MCC, and in the 50-fold cell means.

Verdict
  EXACT     all per-fold differences are 0 (expected with the versions in requirements_rev1.txt)
  WITHIN    per-fold differences exist but every cell-mean MCC and AUC differs by <= 0.005
  DIFFERENT any cell mean differs by more than 0.005; report the label and cells to the authors
"""
import argparse, glob, os, sys
import pandas as pd

KEY = ["dataset", "repeat", "fold", "approach", "classifier"]
METRICS = ["mcc", "auc", "f1", "train_mcc"]
TOL = 0.005


def compare(new, ref):
    m = new.merge(ref, on=KEY, suffixes=("_new", "_ref"))
    if len(m) != len(new):
        return dict(rows=len(new), matched=len(m), verdict="DIFFERENT (rows do not match reference)")
    out = dict(rows=len(new), matched=len(m))
    for c in METRICS:
        out[f"max_abs_{c}"] = float((m[c + "_new"] - m[c + "_ref"]).abs().max())
    g = m.groupby(["dataset", "approach", "classifier"])
    cell = max(float((g[c + "_new"].mean() - g[c + "_ref"].mean()).abs().max()) for c in ("mcc", "auc"))
    out["max_abs_cell_mean"] = cell
    exact = all(out[f"max_abs_{c}"] == 0 for c in METRICS)
    out["verdict"] = "EXACT" if exact else ("WITHIN" if cell <= TOL else "DIFFERENT")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--reference", default="results_rev1")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(f"{a.rerun}/folds_*.csv")):
        label = os.path.basename(path)[6:-4]
        ref_path = f"{a.reference}/folds_{label}.csv"
        if not os.path.exists(ref_path):
            print(f"{label}: no reference file, skipped"); continue
        r = compare(pd.read_csv(path), pd.read_csv(ref_path)); r["label"] = label; rows.append(r)
    if not rows:
        sys.exit("no folds_*.csv files found in --rerun")
    res = pd.DataFrame(rows).set_index("label")
    pd.set_option("display.width", 200); print(res.to_string())
    res.to_csv(f"{a.rerun}/verification_report.csv")
    print("\nreport written to", f"{a.rerun}/verification_report.csv")


if __name__ == "__main__":
    main()
