import numpy as np
from pyhctsa.FeatureCalculator.calculator import FeatureCalculator
from pyhctsa.Utilities.utils import get_dataset
import pickle as pkl
import os, json, hashlib, platform, sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
np.random.seed(0)

WORKSPACE = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd()))
OUTDIR = WORKSPACE / "ci_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

e1000 = get_dataset()
dat = [e1000[0], e1000[100], e1000[500], e1000[890]]
calc = FeatureCalculator()
res = calc.extract(dat)

pkl_path = OUTDIR / "results.pkl"
res.to_pickle(pkl_path)

meta = {
    "rows": int(res.shape[0]),
    "cols": int(res.shape[1]),
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    },
}
(OUTDIR / "meta.json").write_text(json.dumps(meta, indent=2))

print("WROTE:", pkl_path)
print("META:", (OUTDIR / "meta.json"))
