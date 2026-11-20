#!/usr/bin/env python3
"""Run the revision-1 experiments (CMC submission 96252).

Each named configuration writes three files to OUT:
  folds_<label>.csv      one row per dataset x repeat x fold x approach x classifier
  selection_<label>.csv  one row per dataset x repeat x fold x approach (selected metrics, k, timing)
  run_<label>.json       the configuration, the cleaning log with data SHA-256, and wall time

Examples
  python run_experiments.py --data data --out results_rev1 --configs main
  python run_experiments.py --data data --out results_rev1 --configs all
  python run_experiments.py --data data --out quickcheck --quick      # 3 folds, about 1 minute

Configurations that already have a folds file in OUT are skipped, so an interrupted
run can be restarted with the same command.
"""
import argparse, json, os, time
import pandas as pd
import sdp_experiment_v2 as V

ALL = ("none", "inconsistent", "geo", "infotheory")
TREES = ("DT", "RF", "XGBoost")
CONFIGS = {
    # main analysis: nested selection, exact duplicates removed, inner DT objective
    "main":             dict(approaches=ALL, classifiers=TREES + ("SVM",)),
    # the v03 information-gain code path (ranking on the whole outer-training fold), same folds
    "v03ref_ig":        dict(approaches=("infotheory",), classifiers=TREES + ("SVM",), nested=False),
    # cleaning sensitivity: exact duplicates kept
    "nodedup":          dict(approaches=ALL, classifiers=TREES, dedup=False),
    "nodedup_svm_none": dict(approaches=("none",), classifiers=("SVM",), dedup=False),
    # selector objective matched to the downstream classifier (inner RF/XGBoost use 100 trees)
    "matched_ig_RF":    dict(approaches=("infotheory",), classifiers=("RF",), inner_clf="RF"),
    "matched_ig_XGB":   dict(approaches=("infotheory",), classifiers=("XGBoost",), inner_clf="XGBoost"),
    "matched_geo_XGB":  dict(approaches=("geo",), classifiers=("XGBoost",), inner_clf="XGBoost"),
    # selector settings
    "ig_bins5":         dict(approaches=("infotheory",), classifiers=TREES, ig_bins=5),
    "ig_bins20":        dict(approaches=("infotheory",), classifiers=TREES, ig_bins=20),
    "ig_k5":            dict(approaches=("infotheory",), classifiers=TREES, fixed_k=5),
    "ig_k10":           dict(approaches=("infotheory",), classifiers=TREES, fixed_k=10),
    "ig_k15":           dict(approaches=("infotheory",), classifiers=TREES, fixed_k=15),
    "geo_p20_i50":      dict(approaches=("geo",), classifiers=TREES, geo_pop=20, geo_iters=50),
}
QUICK = {("CM1", 0, 0), ("CM1", 2, 5), ("KC1", 4, 9)}   # used by --quick and verify_rerun.py --quick


def _one(ds, X, y, names, rep, f, tr, te, cfg):
    return V.run_fold(ds, X, y, names, rep, f, tr, te, cfg)


def run_cfg(label, data_dir, out, n_jobs, quick=False):
    from joblib import Parallel, delayed
    cfg = V.Config(label=label, **CONFIGS[label])
    datasets = ["CM1", "KC1"] if quick else V.DATASETS
    data, logs, tasks = V.build_tasks(data_dir, cfg, datasets=datasets)
    if quick:
        tasks = [t for t in tasks if (t[0], t[1], t[2]) in QUICK]
    t0 = time.time()
    if n_jobs == 1:   # plain loop; use on machines where process pools are restricted
        res = [_one(ds, *data[ds], rep, f, tr, te, cfg) for ds, rep, f, tr, te in tasks]
    else:
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one)(ds, *data[ds], rep, f, tr, te, cfg) for ds, rep, f, tr, te in tasks)
    wall = round(time.time() - t0, 1)
    pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(f"{out}/folds_{label}.csv", index=False)
    pd.DataFrame([s for _, sel in res for s in sel]).to_csv(f"{out}/selection_{label}.csv", index=False)
    json.dump(dict(config=cfg.asdict(), logs=logs, wall_seconds=wall, quick=quick),
              open(f"{out}/run_{label}.json", "w"), indent=1, default=str)
    print(f"{label}: {len(tasks)} folds, {wall / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_rev1")
    ap.add_argument("--configs", nargs="+", default=["main"], help="names from CONFIGS, or 'all'")
    ap.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true", help="three folds of the main configuration only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.quick:
        return run_cfg("main", a.data, a.out, a.n_jobs, quick=True)
    names = list(CONFIGS) if a.configs == ["all"] else a.configs
    for label in names:
        if os.path.exists(f"{a.out}/folds_{label}.csv"):
            print(f"{label}: output exists, skipped"); continue
        run_cfg(label, a.data, a.out, a.n_jobs)


if __name__ == "__main__":
    main()
