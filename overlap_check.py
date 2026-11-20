"""Share of test modules whose 21-metric vector also occurs in the training fold.
Run from the package root:  python overlap_check.py data rev1_results/tables/train_test_vector_overlap.csv
Uses the same folds as sdp_experiment_v2.py (seed 42, 5x10 repeated stratified CV)."""
import sys, numpy as np, pandas as pd
import sdp_experiment_v2 as V

def overlap(data_dir):
    rows = []
    for dedup in (False, True):
        data, logs, tasks = V.build_tasks(data_dir, V.Config(dedup=dedup))
        keys = {ds: pd.util.hash_pandas_object(pd.DataFrame(X).fillna(-1e300), index=False).values
                for ds, (X, y, n) in data.items()}
        for ds, rep, f, tr, te in tasks:
            seen = set(keys[ds][tr])
            rows.append(dict(dataset=ds, dedup=dedup,
                             pct=100 * np.mean([k in seen for k in keys[ds][te]])))
    return (pd.DataFrame(rows).groupby(["dataset", "dedup"]).pct.mean()
            .unstack().round(1))

if __name__ == "__main__":
    import os
    out = overlap(sys.argv[1] if len(sys.argv) > 1 else "data")
    path = sys.argv[2] if len(sys.argv) > 2 else "train_test_vector_overlap.csv"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path)
    print(out)
