
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
 
from matlab import engine
import joblib
from pyhctsa.utils import get_dataset, z_score
import yaml
import itertools
import matlab
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, mean_absolute_error
import importlib
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr
import pandas as pd
import argparse
 
try:
    from tqdm import tqdm
    def _progress(it, total=None):
        return tqdm(it, total=total)
except ImportError:
    def _progress(it, total=None):
        return it
 

def _flatten(d, prefix=""):
    """Nested struct -> {dotted_key: scalar}."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat
 
 
def _distribution_parity(py_dist, mat_dist):
    """Compare two per-series seed distributions for one feature."""
    py_dist = np.asarray(py_dist, float); py_dist = py_dist[np.isfinite(py_dist)]
    mat_dist = np.asarray(mat_dist, float); mat_dist = mat_dist[np.isfinite(mat_dist)]
    return {
        "py":  np.mean(py_dist)  if py_dist.size  else np.nan,
        "mat": np.mean(mat_dist) if mat_dist.size else np.nan,
        "std_py":  np.std(py_dist,  ddof=1) if py_dist.size  > 1 else np.nan,
        "std_mat": np.std(mat_dist, ddof=1) if mat_dist.size > 1 else np.nan,
    }
 
 
def generate_test_cases(yaml_file):
    """Expand a YAML config into a flat list of (suite_name, settings, param_set)."""
    yaml.add_constructor("!range", lambda loader, node: range(*loader.construct_sequence(node)))
    yaml.add_constructor("!list_range", lambda loader, node: list(range(*loader.construct_sequence(node))))
 
    with open(yaml_file, "r") as f:
        config = yaml.full_load(f)
 
    cases = []
    for suite_name, settings in config["functions"].items():
        for case in settings["test_cases"]:
            keys, values = zip(*case.items())
            expanded_values = [
                list(v) if isinstance(v, range) else [v]
                for v in values
            ]
            for combination in itertools.product(*expanded_values):
                param_set = dict(zip(keys, combination))
                cases.append((suite_name, settings, param_set))
    return cases
 
 
def _normalize(val):
    """Recursively convert MATLAB types / nested structs to plain Python types."""
    if isinstance(val, (matlab.double, matlab.single, matlab.int32, matlab.int64)):
        val = np.array(val).squeeze()
        return float(val) if val.ndim == 0 else val
    if isinstance(val, dict):
        return {k: _normalize(v) for k, v in val.items()}
    if isinstance(val, (np.ndarray, list)):
        val = np.array(val).squeeze()
        return float(val) if val.ndim == 0 else val
    return val
 
 
def _to_matlab_column(arr):
    """Convert a 1-D array-like into an N×1 matlab.double."""
    col = np.asarray(arr, dtype=float).reshape(-1, 1)
    return matlab.double(col.tolist())
 
 
def _to_matlab_row(arr):
    """Convert a 1-D array-like into a 1×N matlab.double."""
    row = np.asarray(arr, dtype=float).reshape(1, -1)
    return matlab.double(row.tolist())
 
 
def _coerce_scalar(v):
    """Coerce Python scalars to MATLAB-friendly types."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    return v
 
 
def test_matlab_parity(matlab_engine, data, suite_name, settings, params):
    run_args = params.copy()
    should_zscore = run_args.pop("zscore", False)
    should_abs = run_args.pop("abs", False)
 
    test_data = z_score(data) if should_zscore else data.copy()
    if should_abs:
        test_data = np.abs(test_data)
 
    orientation = settings.get("matlab_orientation", "column")
    mat_data = (_to_matlab_row(test_data) if orientation == "row"
                else _to_matlab_column(test_data))
 
    arg_map = settings.get("arg_map", {})
 
    is_stochastic = settings.get("stochastic", False)
    seed_param = settings.get("seed_param", "random_seed")
    n_seeds = settings.get("n_seeds", 30) if is_stochastic else 1
 
    run_args.pop(seed_param, None)
 
    base_name_parts = []
    for k, v in sorted(params.items()):
        if k == seed_param:
            continue
        v_clean = str(v).translate(str.maketrans("", "", " []'"))
        base_name_parts.append(f"{k}{v_clean}")
    base_name = "_".join(base_name_parts)
 
    module = importlib.import_module(f"pyhctsa.operations.{settings['python_module']}")
    op_func = getattr(module, settings["python_func"])
    func_name = settings["matlab_func"]
 
    def _build_positional():
        mat_kwargs = {}
        for pk, pv in run_args.items():
            mat_name = arg_map.get(pk, pk)
            if isinstance(pv, list):
                mat_kwargs[mat_name] = matlab.double(pv)
            else:
                mat_kwargs[mat_name] = _coerce_scalar(pv)
 
        for mat_name, val in settings.get("matlab_extra_args", {}).items():
            if isinstance(val, list):
                mat_kwargs[mat_name] = matlab.double(val)
            else:
                mat_kwargs[mat_name] = _coerce_scalar(val)
 
        positional = [mat_data]
        for arg_name in settings.get("matlab_signature", []):
            if arg_name == "y" or arg_name == arg_map.get(seed_param, seed_param):
                continue
            if arg_name in mat_kwargs:
                positional.append(mat_kwargs[arg_name])
        return positional
 
    def _run_once(seed):
        """One paired evaluation. Returns (py_res|None, mat_res|None, err|None)."""
        py_extra = {seed_param: seed} if seed is not None else {}
        py_res = _normalize(op_func(test_data, **run_args, **py_extra))
 
        if seed is not None:
            matlab_engine.eval(f"rng({int(seed)},'twister');", nargout=0)
        try:
            mat_res = _normalize(getattr(matlab_engine, func_name)(*_build_positional(), nargout=1))
        except matlab.engine.MatlabExecutionError as exc:
            msg = str(exc).strip()
            return py_res, None, (msg.splitlines()[-1] if msg else "MatlabExecutionError")
        return py_res, mat_res, None
 
    results = {}
 
    # ---- deterministic path ----
    if not is_stochastic:
        py_res, mat_res, err = _run_once(None)
        if not isinstance(py_res, dict):
            py_res = {"output": py_res}
 
        def fill_nan(py_obj, key=""):
            if isinstance(py_obj, dict):
                for k, v in py_obj.items():
                    fill_nan(v, f"{key}.{k}" if key else k)
            else:
                results[f"{base_name}::{key}"] = {"py": py_obj, "mat": np.nan}
 
        if err is not None:
            print(f"MATLAB error for {base_name} (recording NaN): {err}")
            fill_nan(py_res)
            return results
 
        if not isinstance(mat_res, dict):
            mat_res = {"output": mat_res}
 
        def flatten_and_pair(py_obj, mat_obj, key=""):
            if isinstance(py_obj, dict) and isinstance(mat_obj, dict):
                for k in set(py_obj) | set(mat_obj):
                    nk = f"{key}.{k}" if key else k
                    if k not in py_obj:
                        results[f"{base_name}::{nk}"] = {"error": "Missing in Python"}
                    elif k not in mat_obj:
                        results[f"{base_name}::{nk}"] = {"error": "Missing in MATLAB"}
                    else:
                        flatten_and_pair(py_obj[k], mat_obj[k], nk)
            else:
                results[f"{base_name}::{key}"] = {"py": py_obj, "mat": mat_obj}
 
        flatten_and_pair(py_res, mat_res)
        return results

    seed_py, seed_mat, last_err = [], [], None
    for s in range(n_seeds):
        py_res, mat_res, err = _run_once(s)
        if err is not None:
            last_err = err
            continue
        if not isinstance(py_res, dict):
            py_res = {"output": py_res}
        if not isinstance(mat_res, dict):
            mat_res = {"output": mat_res}
        seed_py.append(_flatten(py_res))
        seed_mat.append(_flatten(mat_res))
 
    if not seed_py:
        print(f"MATLAB error for {base_name} (all seeds, NaN): {last_err}")
        py_once = op_func(test_data, **run_args)
        py_once = py_once if isinstance(py_once, dict) else {"output": py_once}
        for k in _flatten(_normalize(py_once)):
            results[f"{base_name}::{k}"] = {"py": np.nan, "mat": np.nan}
        return results
 
    all_keys = set().union(*(set(d) for d in seed_py), *(set(d) for d in seed_mat))
    for k in all_keys:
        py_dist  = [d[k] for d in seed_py  if k in d]
        mat_dist = [d[k] for d in seed_mat if k in d]
        if not py_dist:
            results[f"{base_name}::{k}"] = {"error": "Missing in Python"}
        elif not mat_dist:
            results[f"{base_name}::{k}"] = {"error": "Missing in MATLAB"}
        else:
            results[f"{base_name}::{k}"] = _distribution_parity(py_dist, mat_dist)
    return results
 
 
def _safe_correlation(py, mat, atol=1e-8):
    """Pearson r with fallbacks for the degenerate cases common in TS features."""
    if py.size < 2:
        return np.nan
    if np.allclose(py, mat, atol=atol):
        return 1.0
    try:
        r, _ = pearsonr(py, mat)
    except ValueError:
        return np.nan
    return r
 
def _accumulate(aggregator, res):
    """Fold one test_matlab_parity result dict into a per-param aggregator."""
    for full_key, values in res.items():
        if "error" in values:
            print(f"Skipping {full_key}: {values['error']}")
            continue
        aggregator[full_key]["py"].append(values["py"])
        aggregator[full_key]["mat"].append(values["mat"])
 
 
def _finalize_param(settings, aggregator):
    """Turn one param's cross-series aggregator into (records, raw_outputs)."""
    records, raw_outputs = [], {}
    for feature_name, lists in aggregator.items():
        py_vec  = np.asarray(lists["py"],  dtype=float)
        mat_vec = np.asarray(lists["mat"], dtype=float)
        raw_outputs[feature_name] = {"py": py_vec, "mat": mat_vec}
 
        mask = np.isfinite(py_vec) & np.isfinite(mat_vec)
        py_valid, mat_valid = py_vec[mask], mat_vec[mask]
 
        r = _safe_correlation(py_valid, mat_valid)
        if py_valid.size >= 1:
            medae = median_absolute_error(py_valid, mat_valid)
            mae = mean_absolute_error(py_valid, mat_valid)
            mape = mean_absolute_percentage_error(py_valid, mat_valid)
        else:
            mae = mape = np.nan
 
        fname = feature_name.split(":")[2]
        if fname == 'output':
            fname = feature_name.split(":")[0]
        records.append({
            "python_func": settings["python_func"],
            "params": feature_name.split(":")[0],
            "feature": fname,
            "matlab_func": settings["matlab_func"],
            "python_module": settings["python_module"],
            "corr": r, "mae": mae, "mape": mape,
            "nan_count_py": int(np.sum(~np.isfinite(py_vec))),
            "nan_count_ml": int(np.sum(~np.isfinite(mat_vec))),
        })
        print(f"{settings['python_func']}-{feature_name}: corr={r:.4f}")
    return records, raw_outputs
 
 
def get_correlations(eng, data, params):
    """Serial path (kept as a reference / single-engine fallback)."""
    records, raw_outputs = [], {}
    for suite_name, settings, param_dict in params:
        aggregator = defaultdict(lambda: {"py": [], "mat": []})
        for series in data:
            res = test_matlab_parity(eng, series, suite_name, settings, param_dict)
            _accumulate(aggregator, res)
        recs, raws = _finalize_param(settings, aggregator)
        records.extend(recs)
        raw_outputs.update(raws)
    return pd.DataFrame(records), raw_outputs

def _shared_noise_parity(eng, series, suite_name, settings, params):
    """Paired CO_AddNoise-style parity with an injected, shared noise vector.

    Reproduces MATLAB's internal draw (BF_ResetSeed(seed); randn(size(y))) and
    feeds the same vector into the Python op, so the only remaining difference
    is the algorithm itself. Returns the same {base::key: {py, mat}} dict shape
    as the deterministic branch of test_matlab_parity, so _accumulate works.
    """
    run_args = params.copy()
    should_zscore = run_args.pop("zscore", False)
    should_abs = run_args.pop("abs", False)
    seed_param = settings.get("seed_param", "random_seed")
    run_args.pop(seed_param, None)

    test_data = z_score(series) if should_zscore else series.copy()
    if should_abs:
        test_data = np.abs(test_data)

    orientation = settings.get("matlab_orientation", "row")  # CO_AddNoise wants a row
    mat_data = (_to_matlab_row(test_data) if orientation == "row"
                else _to_matlab_column(test_data))

    # base_name exactly as in test_matlab_parity (so keys line up downstream)
    base_name = "_".join(
        f'{k}{str(v).translate(str.maketrans("", "", " []\x27"))}'
        for k, v in sorted(params.items()) if k != seed_param)

    seed_val = settings.get("noise_seed", "default")
    n = float(len(test_data))

    eng.BF_ResetSeed(seed_val, nargout=0)
    if orientation == "row":
        noise = np.asarray(eng.randn(1.0, n)).reshape(-1)
    else:
        noise = np.asarray(eng.randn(n, 1.0)).reshape(-1)

    # seed goes in its signature slot so the internal draw matches
    arg_map = settings.get("arg_map", {})
    mat_seed_name = arg_map.get(seed_param, seed_param)
    mat_kwargs = {}
    for pk, pv in run_args.items():
        name = arg_map.get(pk, pk)
        mat_kwargs[name] = matlab.double(pv) if isinstance(pv, list) else _coerce_scalar(pv)
    for name, val in settings.get("matlab_extra_args", {}).items():
        mat_kwargs[name] = matlab.double(val) if isinstance(val, list) else _coerce_scalar(val)

    positional = [mat_data]
    for arg_name in settings.get("matlab_signature", []):
        if arg_name == "y":
            continue
        if arg_name == mat_seed_name:
            positional.append(seed_val)
        elif arg_name in mat_kwargs:
            positional.append(mat_kwargs[arg_name])

    func_name = settings["matlab_func"]
    module = importlib.import_module(f"pyhctsa.operations.{settings['python_module']}")
    op_func = getattr(module, settings["python_func"])

    # inject the same noise (random_seed is ignored once noise is set)
    noise_param = settings.get("noise_param", "noise")
    py_extra = {seed_param: 0, noise_param: noise}
    py_res = _normalize(op_func(test_data, **run_args, **py_extra))
    if not isinstance(py_res, dict):
        py_res = {"output": py_res}

    results = {}
    try:
        mat_res = _normalize(getattr(eng, func_name)(*positional, nargout=1))
    except matlab.engine.MatlabExecutionError as exc:
        msg = str(exc).strip()
        err = msg.splitlines()[-1] if msg else "MatlabExecutionError"
        print(f"MATLAB error for {base_name} (recording NaN): {err}")
        for k in _flatten(py_res):
            results[f"{base_name}::{k}"] = {"py": _flatten(py_res)[k], "mat": np.nan}
        return results

    if not isinstance(mat_res, dict):
        mat_res = {"output": mat_res}

    def flatten_and_pair(py_obj, mat_obj, key=""):
        if isinstance(py_obj, dict) and isinstance(mat_obj, dict):
            for k in set(py_obj) | set(mat_obj):
                nk = f"{key}.{k}" if key else k
                if k not in py_obj:
                    results[f"{base_name}::{nk}"] = {"error": "Missing in Python"}
                elif k not in mat_obj:
                    results[f"{base_name}::{nk}"] = {"error": "Missing in MATLAB"}
                else:
                    flatten_and_pair(py_obj[k], mat_obj[k], nk)
        else:
            results[f"{base_name}::{key}"] = {"py": py_obj, "mat": mat_obj}

    flatten_and_pair(py_res, mat_res)
    return results

def get_correlations_shared_noise(eng, data, params):
    """Serial shared-noise parity. Returns (df, raw_outputs) in the main-table schema."""
    records, raw_outputs = [], {}
    for suite_name, settings, param_dict in params:
        aggregator = defaultdict(lambda: {"py": [], "mat": []})
        for series in data:
            res = _shared_noise_parity(eng, series, suite_name, settings, param_dict)
            _accumulate(aggregator, res)
        recs, raws = _finalize_param(settings, aggregator)
        records.extend(recs)
        raw_outputs.update(raws)
    return pd.DataFrame(records), raw_outputs
 
 
_ENGINE = None
_DATA = None
_PARAMS = None
 
 
def _init_worker(hctsa_path, jidt_path, dataset_name, yaml_file):
    """Boot a dedicated MATLAB engine + load data/params once per worker."""
    global _ENGINE, _DATA, _PARAMS
    eng = engine.start_matlab()
    eng.addpath(eng.genpath(hctsa_path), nargout=0)
    eng.javaaddpath(jidt_path, nargout=0)
    _ENGINE = eng
    _DATA = get_dataset(dataset_name)
    _PARAMS = generate_test_cases(yaml_file)
 
 
def _run_task(task):
    """One (param, series) parity evaluation on this worker's engine."""
    param_idx, series_idx = task
    suite_name, settings, param_dict = _PARAMS[param_idx]
    series = _DATA[series_idx]
    res = test_matlab_parity(_ENGINE, series, suite_name, settings, param_dict)
    return param_idx, res

def _run_task_shared_noise(task):
    """One (param, series) shared-noise parity evaluation on this worker's engine."""
    param_idx, series_idx = task
    suite_name, settings, param_dict = _PARAMS[param_idx]
    series = _DATA[series_idx]
    res = _shared_noise_parity(_ENGINE, series, suite_name, settings, param_dict)
    return param_idx, res

def _parallel_driver(task_fn, hctsa_path, jidt_path, dataset_name, yaml_file,
                     max_workers=None):
    """Shared parallel engine: boots one MATLAB engine per worker and folds
    every (param, series) result into per-param aggregators. `task_fn` is a
    module-level callable (picklable) taking (param_idx, series_idx) -> (param_idx, res)."""
    params = generate_test_cases(yaml_file)
    data = get_dataset(dataset_name)
    n_params, n_series = len(params), len(data)

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    tasks = [(pi, si) for pi in range(n_params) for si in range(n_series)]
    aggregators = {pi: defaultdict(lambda: {"py": [], "mat": []})
                   for pi in range(n_params)}

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(hctsa_path, jidt_path, dataset_name, yaml_file),
    ) as ex:
        futures = [ex.submit(task_fn, t) for t in tasks]
        for fut in _progress(as_completed(futures), total=len(futures)):
            try:
                param_idx, res = fut.result()
            except Exception as exc:
                print(f"[worker error] {exc}")
                continue
            _accumulate(aggregators[param_idx], res)

    records, raw_outputs = [], {}
    for pi in range(n_params):
        _, settings, _ = params[pi]
        recs, raws = _finalize_param(settings, aggregators[pi])
        records.extend(recs)
        raw_outputs.update(raws)
    return pd.DataFrame(records), raw_outputs

def get_correlations_parallel(hctsa_path, jidt_path, dataset_name, yaml_file,
                              max_workers=None):
    """Parallel drop-in for get_correlations. Returns (df, raw_outputs)."""
    return _parallel_driver(_run_task, hctsa_path, jidt_path, dataset_name,
                            yaml_file, max_workers=max_workers)


def get_correlations_shared_noise_parallel(hctsa_path, jidt_path, dataset_name, yaml_file,
                                           max_workers=None):
    """Parallel shared-noise parity. Returns (df, raw_outputs)
    in the main-table schema. The YAML here should contain ONLY noise-injectable ops."""
    return _parallel_driver(_run_task_shared_noise, hctsa_path, jidt_path, dataset_name,
                            yaml_file, max_workers=max_workers)

def main():
    parser = argparse.ArgumentParser(description="Run pyhctsa numerical-parity validation against MATLAB HCTSA.")

    parser.add_argument("--config", required=True, help="Path to validation YAML config (e.g., validate_deterministic/.yaml)")
    parser.add_argument("--dataset", default='e1000', help='Dataset name passed to get_dataset (default: e1000)')
    parser.add_argument("--mode", choices=['deterministic', 'shared-noise'], default='deterministic', help="Validation path to run (default: deterministic).")
    parser.add_argument("--output", default=None, help="Output .pkl path (default: validation_<mode>.pkl).")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Number of parallel MATLAB-engine workers.")
    parser.add_argument("--hctsa-path",
                        default=os.environ.get("HCTSA_PATH"),
                        help="Path to hctsa (or set HCTSA_PATH env var).")
    parser.add_argument("--jidt-path",
                        default=os.environ.get("JIDT_PATH"),
                        help="Path to infodynamics.jar (or set JIDT_PATH env var).")
    args = parser.parse_args()
    if not args.hctsa_path or not args.jidt_path:
        parser.error("hctsa and jidt paths are required "
                     "(pass --hctsa-path/--jidt-path or set HCTSA_PATH/JIDT_PATH).")
    driver = (get_correlations_parallel if args.mode == "deterministic"
              else get_correlations_shared_noise_parallel)
    df, _ = driver(
        args.hctsa_path, args.jidt_path, args.dataset, args.config,
        max_workers=args.max_workers,
    )
    out = args.output or f"validation_{args.mode.replace('-', '_')}.pkl"
    df.to_pickle(out)
    print(f"Wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    main()
    # HCTSA_PATH =  '/Users/jmoo2880/Documents/hctsa' #-> REPLACE WITH ACTUAL LOCATION
    # JIDT_PATH =  '/Users/jmoo2880/Documents/hctsa/Toolboxes/infodynamics-dist/infodynamics.jar' #-> REPLACE WITH ACTUAL LOCATION

