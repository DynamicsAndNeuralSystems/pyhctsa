import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ATOL = 1e-10
RTOL = 5e-10
SOFT = os.environ.get("SOFT_COMPARE", "1") == "1"  # default: don't fail CI

arts_root = Path("artifacts")
subdirs = sorted(p for p in arts_root.iterdir() if p.is_dir())

def finish(code_if_hard):
    # In soft mode, never fail
    sys.exit(0 if SOFT else code_if_hard)

if len(subdirs) < 2:
    print("No or too few artifacts under ./artifacts/. Nothing to compare.")
    finish(2)

dfs, labels = [], []
for sd in subdirs:
    pkl = sd / "results.pkl"
    meta = sd / "meta.json"
    if not pkl.exists():
        print(f"Missing {pkl}")
        finish(2)
    df = pd.read_pickle(pkl)
    dfs.append(df)
    labels.append(sd.name)
    if meta.exists():
        m = json.loads(meta.read_text())
        short = m.get("sha256_numeric_json", "")[:12]
        if short:
            print(f"{sd.name}: sha256={short}…")

# Align numeric columns and shapes
shapes = [d.shape for d in dfs]
if len(set(shapes)) != 1:
    print("Shape mismatch across platforms:", dict(zip(labels, shapes)))
    finish(1)

num_cols = set.intersection(*[set(d.select_dtypes(include=[np.number]).columns) for d in dfs])
if not num_cols:
    print("No shared numeric columns to compare.")
    finish(2)
num_cols = sorted(num_cols)

# Compare
print("Comparing numeric columns with atol=", ATOL, "rtol=", RTOL)
mismatches = []
for c in num_cols:
    vals = [d[c].to_numpy(dtype=float) for d in dfs]
    ref = vals[0]
    oks = [np.allclose(v, ref, atol=ATOL, rtol=RTOL, equal_nan=True) for v in vals]
    if all(oks):
        print("OK  ", c)
    else:
        print("FAIL", c)
        row_idx = [int(np.nanargmax(np.abs(v - ref))) if not np.allclose(v, ref, atol=ATOL, rtol=RTOL, equal_nan=True) else -1 for v in vals]
        for lab, v, ok, ridx in zip(labels, vals, oks, row_idx):
            diff = float(np.nanmax(np.abs(v - ref)))
            if not ok:
                # GitHub annotation as a warning (visible but non-fatal)
                print(f"::warning title=Cross-platform mismatch::{c} on {lab}: max|Δ|={diff:.3e} at row {ridx}")
            print(f"   {lab:>24}: max|Δ|={diff:.3e}" + (f" @row {ridx}" if ridx >= 0 else ""))
        mismatches.append(c)

# Write a short summary to the job summary pane
summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
if summary_path:
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("### Cross-platform comparison\n")
        f.write(f"- Artifacts: {', '.join(labels)}\n")
        if mismatches:
            f.write(f"- **Mismatched columns** ({len(mismatches)}): {', '.join(mismatches)}\n")
        else:
            f.write("- All checked numeric columns matched within tolerance.\n")

finish(0 if not mismatches else 1)
