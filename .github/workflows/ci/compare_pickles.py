import sys, json
import pickle as pkl 
from pathlib import Path
import pandas as pd
import numpy as np

ATOL = 1e-10
RTOL = 5e-10

# load in the pickle files
artifacts_root = Path("artifacts")
subdirs = sorted(p for p in artifacts_root.iterdir() if p.is_dir())
if len(subdirs) < 2:
    print("Need at least two artifact directories under ./artifacts/", file=sys.stderr)
    sys.exit(2)

dfs, labels = [], []
for sd in subdirs:
    pkl = sd / "results.pkl"
    meta = sd / "meta.json"
    if not pkl.exists():
        print(f"Missing {pkl}", file=sys.stderr); sys.exit(2)
    df = pd.read_pickle(pkl)
    dfs.append(df)
    labels.append(sd.name)
    if meta.exists():
        print(sd.name, "->", json.loads(meta.read_text()).get("sha256_numeric_json", "")[:12], "...")

num_cols = set.intersection(*[set(d.select_dtypes(include=[np.number]).columns) for d in dfs])
if not num_cols:
    print("No shared numeric columns to compare.", file=sys.stderr)
    sys.exit(2)
num_cols = sorted(num_cols)

shapes = [d.shape for d in dfs]
if len(set(shapes)) != 1:
    print("Different shapes across platforms:", dict(zip(labels, shapes)), file=sys.stderr)
    sys.exit(1)

ok_all = True
for c in num_cols:
    vals = [d[c].to_numpy(dtype=float) for d in dfs]
    ref = vals[0]
    all_ok = all(np.allclose(v, ref, atol=ATOL, rtol=RTOL, equal_nan=True) for v in vals[1:])
    print(("OK  " if all_ok else "FAIL"), c)
    if not all_ok:
        # print a concise summary to help debug
        for lab, v in zip(labels, vals):
            diff = np.nanmax(np.abs(v - ref))
            print(f"   {lab:>24}: max|Δ|={diff:.3e}")
    ok_all &= all_ok

sys.exit(0 if ok_all else 1)
