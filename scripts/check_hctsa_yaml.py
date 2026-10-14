#!/usr/bin/env python
"""Compare pyhctsa's default feature set with hctsa's.

pyhctsa's ``configurations/hctsa.yaml`` should register, for every hctsa function pyhctsa implements, the same
calls (parameter settings) as hctsa's ``FeatureSets/INP_mops_hctsa.txt`` (master operations) and
``INP_ops_hctsa.txt`` (operations: one master output field each). This script reads a hctsa checkout and reports

(a) hctsa calls of implemented functions that no pyhctsa config reproduces (the functions pyhctsa implements are those
    with a ``legacy_name`` in the yaml, plus unregistered public functions of ``pyhctsa/operations`` whose name matches
    an hctsa function, or is listed in ``EXTRA_IMPL``);
(b) pyhctsa configs that match no hctsa call (a function deregistered in hctsa, or a differing setting);
(c) output-name mismatches: a call's hctsa fields (``master.field`` of INP_ops) against the keys pyhctsa returns, found
    by running each matched config on test series (``--run``); pyhctsa-only keys are left out of the feature set with
    the config's ``exclude:`` (or ``select:``);
(d) hctsa functions pyhctsa does not implement; python arguments with no hctsa counterpart; calls matched only by
    ignoring explicitly set hctsa arguments that pyhctsa lacks;
(e) feature labels two configs share (the calculator keeps only the later one).

A pyhctsa config matches an hctsa call when the input agrees (see ``INPUTS``) and every argument they share has the same
value. An argument a call leaves out takes hctsa's default (read from the ``if nargin < k`` blocks of the .m file; if
that cannot be read, pyhctsa's default is assumed to be the same); a pyhctsa default of None means "the function's own
default" and matches a left-out hctsa argument. pyhctsa arguments with no hctsa counterpart must sit at their default.
Arguments are paired by name (camelCase against snake_case) and by the ``ARGMAP`` table below; ``VALUES``,
``REQUIRE`` and ``PARAM_FORMS`` cover options spelled differently.

Usage (from a pyhctsa checkout; the pyhctsa package must be importable)::

    python scripts/check_hctsa_yaml.py /path/to/hctsa                  # report (a), (b), (d), (e)
    python scripts/check_hctsa_yaml.py /path/to/hctsa --run            # also (c): runs every matched config (~1 min)
    python scripts/check_hctsa_yaml.py /path/to/hctsa --suggest        # yaml for the missing calls of (a)
    python scripts/check_hctsa_yaml.py /path/to/hctsa --suggest --run  # ... and the exclude: lists for (c)
    python scripts/check_hctsa_yaml.py /path/to/hctsa --json out.json  # everything, machine-readable

Exit status is 1 if anything is reported under (a), (b), (c) or (e), else 0. When hctsa changes its registrations,
re-run it: new calls of ported functions show up under (a) (``--suggest`` writes the configs), removed ones under (b);
a function whose arguments or option values do not pair up by name needs an entry in ``ARGMAP`` / ``VALUES``.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------------------------------------------
# Conventions that tie pyhctsa to hctsa. Extend these when a new function or option does not pair up by name.
# ---------------------------------------------------------------------------------------------------------------
# hctsa input expression -> (zscore, abs, preprocess) of a pyhctsa config
INPUTS = {
    "x": (False, False, None), "y": (False, False, None),
    "x_z": (True, False, None),
    "abs(x_z)": (True, True, None),
    "diff(x_z)": (True, False, "diff1"),
    "zscore(abs(x_z))": (True, False, "zscore_abs"),
    "zscore(sign(x_z))": (True, False, "zscore_sign"),
    "zscore(BF_PreProcess(x_z,'decimate_ac1e'))": (True, False, "decimate_ac1e"),
}
# unregistered pyhctsa functions with a name that does not follow hctsa's: hctsa function -> (module, python function)
EXTRA_IMPL = {
    "EN_BubbleEn": ("entropy", "bubble_entropy"), "EN_DispEn": ("entropy", "dispersion_entropy"),
    "EN_FuzzyEn": ("entropy", "fuzzy_entropy"), "EN_PermEnComplexity": ("entropy", "permutation_entropy_complexity"),
    "EN_wentropy": ("entropy", "wavelet_entropy"), "NL_c1": ("nonlinearity", "tisean_c1"),
    "PP_Iterate": ("pre_process", "preproc_iterate"), "PP_ModelFit": ("pre_process", "preproc_model_fit"),
    "PP_SchreiberDenoise": ("pre_process", "preproc_schreiber_denoise"),
    "SY_StdNthDerChange": ("stationarity", "std_nth_deriv_change"), "SY_nstat_z": ("stationarity", "nstat_z"),
    "SY_DriftingMeanCUSUM": ("stationarity", "drifting_mean_cusum"),
    "ST_PeakIntervals": ("stationarity", "peak_intervals"),
    "SD_Surrogates": ("surrogates", "surrogates"), "CP_WaveletVarChg": ("changepoint", "wavelet_var_chg"),
    "EX_ExtremeEventOrder": ("extreme_events", "extreme_event_order"),
}
# python parameter -> hctsa argument, where the two do not pair up by name: {hctsa function: {python param: hctsa arg}};
# a (arg, k) value is the k-th element of a cell/vector argument (pyhctsa splits hctsa's embedParams = {tau, m})
ARGMAP: dict[str, dict] = {
    "CO_FirstMin": {"min_what": "minWhat", "max_what": "minWhat"},
    "CO_NonlinearAutocorr": {"absval": "doAbs"},
    "CR_RAD": {"centre": "doAbs"},
    "DN_TrimmedMean": {"p_exclude": "n"},
    "EN_LZComplexity": {"n_bits": "n"},
    "MF_hmm_CompareNStates": {"n_states": "nstater"},
    "SY_StatAv": {"extra_param": "n"},
    "SY_StdNthDer": {"ndr": "n"},
    "PP_Compare": {"detrend_meth": "detrndmeth"},
    "EN_Shannon": {"num_bins": "numBin"},
    "SB_BinaryGapHomogeneity": {"stretch_what": "gapWhat"},
    "NL_PoincareSection": {"tau": ("embedParams", 0)},
    "NL_LocalDensity": {"tau": ("embedParams", 0), "m": ("embedParams", 1)},
}
# one pyhctsa function standing for one setting of an hctsa argument: (hctsa function, python function) ->
# (hctsa argument, required value, the value when the call leaves it out)
REQUIRE = {
    ("CO_FirstMin", "first_min"): ("minNotMax", True, True),
    ("CO_FirstMin", "first_max"): ("minNotMax", False, True),
}
# python option value -> hctsa value, per (hctsa function, hctsa argument)
VALUES: dict[tuple, dict] = {
    ("CO_FirstCrossing", "threshold"): {"1/e": 1 / math.e},
    ("CO_AutoCorrShape", "stopWhen"): {"pos_drown": "posDrown"},      # pyhctsa still spells these in snake_case
    ("CO_AutoCorrX2Shape", "maxLag"): {"double_drown": "doubleDrown"},
}
# arguments that only steer the random-number generator or plotting: left out of the comparison
IGNORE = {"randomseed", "seed", "doplot", "beverbose", "randomstate", "rng"}


# ---------------------------------------------------------------------------------------------------------------
# hctsa side
# ---------------------------------------------------------------------------------------------------------------
def split_args(s: str) -> list[str]:
    """Split a MATLAB argument list at its top-level commas (brackets, braces, parentheses and quotes nest)."""
    out, depth, cur, quote = [], 0, [], False
    for i, ch in enumerate(s):
        if quote:
            cur.append(ch)
            if ch == "'":
                quote = False
            continue
        if ch == "'" and (i == 0 or s[i - 1] in "(,[{ \t"):   # an opening quote, not a transpose
            quote = True
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur or out:
        out.append("".join(cur).strip())
    return out


class Call:
    """One master operation of hctsa: ``Func(input,arg,...)`` with its label."""

    def __init__(self, code: str, label: str):
        self.code, self.label = code, label
        m = re.fullmatch(r"(\w+)\((.*)\)", code.strip())
        if not m:
            raise ValueError(f"cannot parse call {code!r}")
        self.func = m.group(1)
        parts = split_args(m.group(2))
        self.input, self.args = parts[0], parts[1:]


def parse_mops(path: Path) -> list[Call]:
    calls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"(\S.*\))\s+(\S+)", line)
        if not m:
            raise ValueError(f"cannot parse INP_mops line {line!r}")
        calls.append(Call(m.group(1), m.group(2)))
    return calls


def parse_ops(path: Path, labels: set[str]) -> dict[str, list[tuple[str, str]]]:
    """master label -> [(field, hctsa feature name)] from INP_ops (lines ``master.field  feature_name  keywords``;
    field '' = the call's single output)."""
    out = defaultdict(list)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2 or line.startswith("#"):
            continue
        master, name = parts[0].strip(), parts[1].strip()
        if master in labels:
            out[master].append(("", name))
            continue
        label, _, field = master.rpartition(".")
        while label and label not in labels:       # field names may themselves contain dots
            label, _, f2 = label.rpartition(".")
            field = f"{f2}.{field}"
        if not label:
            raise ValueError(f"INP_ops feature {name}: master {master!r} is not in INP_mops")
        out[label].append((field, name))
    return out


def hctsa_signature(hctsa: Path, func: str) -> list[str] | None:
    """The argument names (after the series) of ``Operations/<func>.m``."""
    p = hctsa / "Operations" / f"{func}.m"
    if not p.exists():
        found = list((hctsa / "Toolboxes").rglob(f"{func}.m"))
        if not found:
            return None
        p = found[0]
    txt = p.read_text(errors="replace")
    m = re.search(rf"^\s*function\s+(?:\[?[^=\n]*\]?\s*=\s*)?{re.escape(func)}\s*\(([^)]*)\)", txt, re.M)
    if not m:
        return None
    return [a.strip() for a in m.group(1).split(",")][1:]


def hctsa_defaults(hctsa: Path, func: str) -> dict:
    """Best-effort defaults of ``Operations/<func>.m``: the ``name = value;`` lines of its ``if nargin < k ...`` blocks
    (values that are not a plain MATLAB literal come back as ("raw", source) and are never compared)."""
    p = hctsa / "Operations" / f"{func}.m"
    if not p.exists():
        found = list((hctsa / "Toolboxes").rglob(f"{func}.m"))
        if not found:
            return {}
        p = found[0]
    out, inblock = {}, False
    for line in p.read_text(errors="replace").splitlines():
        code = line.split("%")[0].strip() if "'%" not in line else line.strip()
        if re.match(r"if\s*\(?\s*nargin\s*<", code):
            inblock = True
            m = re.search(r"\)\s*,?\s*(\w+)\s*=\s*([^;]+);", code)   # a one-line block
            if m:
                out.setdefault(m.group(1), mvalue(m.group(2)))
                inblock = False
            continue
        if inblock:
            if code.startswith("end"):
                inblock = False
                continue
            m = re.fullmatch(r"(\w+)\s*=\s*([^;]+);.*", code)
            if m:
                out.setdefault(m.group(1), mvalue(m.group(2)))
    return out


def mvalue(s: str):
    """A MATLAB argument's source as a Python value: numbers -> float, 'text' -> str, true/false -> bool, [..] and
    {..} -> list, a:b and a:s:b -> list, [] -> None (the default). Anything else -> ("raw", source), which never matches."""
    s = s.strip()
    if s in ("[]", "{}", ""):
        return None
    if s in ("true", "false"):
        return s == "true"
    if re.fullmatch(r"'(?:[^']|'')*'", s):
        return s[1:-1].replace("''", "'")
    if s[0] in "[{" and s[-1] in "]}":
        inner = s[1:-1].strip()
        parts = split_args(inner) if "," in inner or s[0] == "{" else inner.split()
        return [mvalue(p) for p in parts]
    m = re.fullmatch(r"(-?[\d.]+)\s*:\s*(-?[\d.]+)(?:\s*:\s*(-?[\d.]+))?", s)
    if m:
        a, b, c = (float(g) if g else None for g in m.groups())
        step, stop = (b, c) if c is not None else (1.0, b)
        n = int(math.floor((stop - a) / step + 1e-9)) + 1
        return [round(a + k * step, 10) for k in range(n)]
    try:
        return float(s)
    except ValueError:
        pass
    if s in ("Inf", "inf"):
        return math.inf
    if s == "-Inf":
        return -math.inf
    m = re.fullmatch(r"([\d.]+)/exp\((\d+)\)", s)
    if m:
        return float(m.group(1)) / math.exp(float(m.group(2)))
    return ("raw", s)


def norm(v):
    """A comparable form: numbers (and bools) as rounded floats, text lowercased, one-element lists as their element."""
    if isinstance(v, tuple) and v[:1] == ("raw",):
        return v
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return round(float(v), 9)
    if isinstance(v, str):
        if v == "":
            return None   # empty means "the default" in hctsa
        try:
            return round(float(v), 9)   # hctsa passes some numbers as text, e.g. '4'
        except ValueError:
            return v.lower()
    if isinstance(v, (list, tuple, range)):
        v = [norm(x) for x in v]
        return v[0] if len(v) == 1 else tuple(v)
    return v


def _key(name: str) -> str:
    return name.replace("_", "").lower()


# ---------------------------------------------------------------------------------------------------------------
# pyhctsa side
# ---------------------------------------------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    from pyhctsa import calculator   # registers the !range constructor on yaml.SafeLoader
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _default_value(d):
    """A parameter default from its AST node: literals and np.inf / np.nan; anything else -> ("raw", source)."""
    if d is None:
        return None
    try:
        return ast.literal_eval(d)
    except ValueError:
        pass
    neg = isinstance(d, ast.UnaryOp) and isinstance(d.op, ast.USub)
    node = d.operand if neg else d
    consts = {"inf": math.inf, "nan": math.nan, "pi": math.pi}
    if isinstance(node, ast.Attribute) and node.attr in consts or isinstance(node, ast.Name) and node.id in consts:
        v = consts[node.attr if isinstance(node, ast.Attribute) else node.id]
        return -v if neg else v
    return ("raw", ast.unparse(d))


def py_signatures(opdir: Path) -> dict[tuple[str, str], dict]:
    """(module, function) -> {param: default} for the public functions of ``pyhctsa/operations`` (the series excluded)."""
    out = {}
    for p in sorted(opdir.glob("*.py")):
        if p.stem == "__init__":
            continue
        for n in ast.parse(p.read_text()).body:
            if not isinstance(n, ast.FunctionDef) or n.name.startswith("_"):
                continue
            a = n.args
            pos = a.posonlyargs + a.args
            dflt = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
            params = {}
            for arg, d in list(zip(pos, dflt))[1:] + list(zip(a.kwonlyargs, a.kw_defaults)):
                params[arg.arg] = _default_value(d)
            out[(p.stem, n.name)] = params
    return out


def expand(meta: dict):
    """A function's configs as the calculator expands them: (label, kwargs, zscore, abs, preprocess, select, exclude)."""
    from pyhctsa.calculator import _build_label
    base, ordered = meta.get("base_name"), meta.get("ordered_args") or []
    for item in meta.get("configs") or [{}]:
        item = dict(item or {})
        z, ab, pre = item.pop("zscore", False), item.pop("abs", False), item.pop("preprocess", None)
        sel, exc = item.pop("select", None), item.pop("exclude", None)
        keys = list(item)
        for combo in itertools.product(*[v if isinstance(v, list) else [v] for v in item.values()]):
            kw = dict(zip(keys, combo))
            yield _build_label(base, kw, ordered, z, ab, pre), kw, bool(z), bool(ab), pre, sel, exc


def argmap(func: str, params: dict, hargs: list[str]) -> dict:
    """python parameter -> hctsa argument (or (argument, k)): ARGMAP where it names a real hctsa argument, else by
    name (camelCase against snake_case)."""
    if hargs == ["varargin"]:
        return {p: p for p in params}
    known = {_key(a): a for a in hargs}
    out = {}
    for p in params:
        m = ARGMAP.get(func, {}).get(p)
        if m is not None and (m in hargs or (isinstance(m, tuple) and m[0] in hargs)):
            out[p] = m
        elif p in hargs:   # exact: hctsa's d and D differ only in case
            out[p] = p
        elif _key(p) in known and sum(_key(a) == _key(p) for a in hargs) == 1:
            out[p] = known[_key(p)]
        elif "what" + _key(p) in known:   # method <-> whatMethod
            out[p] = known["what" + _key(p)]
    return out


def _cov_canon(v):
    """A GP covariance function in one form: pyhctsa's 'covSEiso_covNoise' shorthand for hctsa's
    {'covSum',{'covSEiso','covNoise'}} (a sum of its parts), else the nested list."""
    if isinstance(v, str) and v.startswith("cov"):
        parts = []
        for t in v.split("_"):
            m = re.fullmatch(r"covMaterniso(\d)", t)   # 'covMaterniso3' is {'covMaterniso', 3}
            parts.append(["covMaterniso", float(m.group(1))] if m else t)
        return ["covSum", parts] if len(parts) > 1 else v
    return v


def _cov_to_python(v):
    """The pyhctsa form of a hctsa covariance function: the shorthand string for a sum of named terms."""
    def term(t):
        if isinstance(t, str):
            return t
        if isinstance(t, list) and len(t) == 2 and t[0] == "covMaterniso":
            return f"covMaterniso{int(t[1])}"
        return None
    if isinstance(v, list) and len(v) == 2 and v[0] == "covSum" and isinstance(v[1], list):
        terms = [term(t) for t in v[1]]
        if None not in terms:
            return "_".join(terms)
    return v


# python parameter -> (canonical form for comparing with hctsa's value, hctsa value -> the form written in the yaml)
PARAM_FORMS = {"cov_func": (_cov_canon, _cov_to_python)}


def given_args(call: Call, hargs: list[str]) -> dict:
    """The arguments a call sets, by hctsa argument name; for a ``varargin`` function, the name/value pairs."""
    if hargs == ["varargin"]:
        return {call.args[i].strip("'"): mvalue(call.args[i + 1]) for i in range(0, len(call.args) - 1, 2)}
    return {hargs[i]: mvalue(a) for i, a in enumerate(call.args) if i < len(hargs)}


def match_one(func, call: Call, hargs, kw, flags, params, amap, name=None, hdef=None):
    """None if the call and the pyhctsa config differ; else the list of hctsa arguments the call sets explicitly that the
    config has no counterpart for (a config that cannot set them is most likely the call leaving them at their defaults)."""
    if INPUTS.get(call.input.replace(" ", "")) != flags:
        return None
    given = given_args(call, hargs)
    req = REQUIRE.get((func, name))
    if req:
        a, want, dflt_when_missing = req
        got = given.get(a)
        if norm(dflt_when_missing if got is None else got) != norm(want):
            return None
    for p, dflt in params.items():
        if _key(p) in IGNORE:
            continue
        pv = kw.get(p, dflt)
        if p in amap:
            a, k = amap[p] if isinstance(amap[p], tuple) else (amap[p], None)
            hv = given.get(a)
            if k is not None:
                hv = hv[k] if isinstance(hv, list) and len(hv) > k else None
            if isinstance(pv, str):
                pv = VALUES.get((func, a), {}).get(pv, pv)
            if p in PARAM_FORMS:
                pv = PARAM_FORMS[p][0](pv)
                hv = PARAM_FORMS[p][0](hv) if hv is not None else hv
            if hv is None:   # left at hctsa's default: its value if known, else assume pyhctsa's default is the same
                if pv is None:   # None is pyhctsa's "use the function's own default"
                    continue
                hd = (hdef or {}).get(a)
                hv = hd if hd is not None and not (isinstance(hd, tuple) and hd[:1] == ("raw",)) else dflt
            if p in PARAM_FORMS:
                hv = PARAM_FORMS[p][0](hv)
            if norm(hv) != norm(pv):
                return None
        elif norm(pv) != norm(dflt):   # a pyhctsa-only option set away from its default
            return None
    if not all(k in params for k in kw):
        return None
    mapped = {m[0] if isinstance(m, tuple) else m for m in amap.values()} | ({req[0]} if req else set())
    return [a for a, v in given.items() if a not in mapped and _key(a) not in IGNORE and v is not None]


# ---------------------------------------------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------------------------------------------
def _flat_keys(r, prefix="") -> list[str]:
    """The column names a returned dict gives (nested dicts join with '.', as the calculator's DataFrame does)."""
    out = []
    for k, v in r.items():
        out += _flat_keys(v, f"{prefix}{k}.") if isinstance(v, dict) else [f"{prefix}{k}"]
    return out


def run_outputs(yaml_path: Path, labels: set[str], series_list) -> dict[str, object]:
    """Run each configured feature on the test series: label -> sorted output keys (None for a scalar, the union over
    the series for a dict) or an 'Error:' text when it failed on every series."""
    import warnings
    from pyhctsa.calculator import FeatureCalculator
    warnings.filterwarnings("ignore")
    fc = FeatureCalculator(str(yaml_path))
    out = {}
    for label, f in fc.feature_funcs.items():
        if label not in labels:
            continue
        keys, errs, scalar = set(), [], False
        for series in series_list:
            try:
                r = f(series)
            except Exception as e:   # noqa: BLE001 - report, do not stop
                errs.append(f"Error: {type(e).__name__}: {e}"[:200])
                continue
            if isinstance(r, dict):
                keys |= set(_flat_keys(r))
            else:
                scalar = True
        out[label] = sorted(keys) if keys else (None if scalar else errs[0])
    return out


def default_series():
    """Test series for --run: an AR(1) plus a sine, and a positive skewed one (some fits need positive values)."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 1000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + e[t]
    x = x + 0.5 * np.sin(2 * np.pi * np.arange(n) / 50)
    return [x, np.exp(x / 2)]


def python_name(v):
    """A hctsa value as yaml-ready Python: whole floats as ints, numbers passed as text ('4') as numbers."""
    if isinstance(v, str):
        try:
            return python_name(float(v))
        except ValueError:
            return v
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return int(v)
    if isinstance(v, list):
        return [python_name(x) for x in v]
    return v


def compare(hctsa: Path, yaml_path: Path, log=print, run=False, series=None) -> dict:
    mops = parse_mops(hctsa / "FeatureSets" / "INP_mops_hctsa.txt")
    ops = parse_ops(hctsa / "FeatureSets" / "INP_ops_hctsa.txt", {c.label for c in mops})
    calls_of = defaultdict(list)
    for c in mops:
        calls_of[c.func].append(c)
    cfg = load_yaml(yaml_path)
    sigs = py_signatures(REPO / "pyhctsa" / "operations")
    lower = {f.lower(): f for f in calls_of}

    impl = {}   # hctsa function -> [(module, python name, meta or None)]
    deregistered = []   # yaml entries whose legacy function is no longer in hctsa's master operations
    no_legacy = []
    for module, fns in cfg.items():
        for name, meta in fns.items():
            leg = (meta.get("legacy_name") or "").strip()
            if not leg:
                no_legacy.append((module, name))
                continue
            if leg not in calls_of:
                hit = lower.get(leg.lower())
                if hit is None:
                    deregistered.append((module, name, leg, meta))
                    continue
                leg = hit
            impl.setdefault(leg, []).append((module, name, meta))
    registered_py = {(m, n) for v in impl.values() for m, n, _ in v} | {(m, n) for m, n, *_ in deregistered} | set(no_legacy)
    unregistered = {}   # hctsa function -> (module, python name): implemented, with no yaml entry
    by_stem = defaultdict(list)
    for (m, n) in sigs:
        by_stem[_key(n)].append((m, n))
    for f in calls_of:
        if f in impl:
            continue
        cand = EXTRA_IMPL.get(f)
        if cand is None:
            hits = [mn for mn in by_stem.get(_key(f.split("_", 1)[1]), []) if mn not in registered_py]
            cand = hits[0] if len(hits) == 1 else None
        if cand and cand in sigs and cand not in registered_py:
            unregistered[f] = cand
    not_implemented = sorted(f for f in calls_of if f not in impl and f not in unregistered)

    res = {"missing": defaultdict(list), "extra": [], "extras_ignored": [], "multi": [], "fields": [],
           "unregistered": {f: list(v) for f, v in unregistered.items()}, "not_implemented": not_implemented,
           "deregistered": [], "counts": {}}
    matched_calls, matched_cfg = set(), set()
    pair_for_label = {}   # yaml label -> hctsa call labels it matches
    all_labels = {}
    for f, entries in list(impl.items()) + [(f, [(m, n, None)]) for f, (m, n) in unregistered.items()]:
        hargs = hctsa_signature(hctsa, f) or []
        hdef = hctsa_defaults(hctsa, f)
        calls = calls_of[f]
        for module, name, meta in entries:
            params = sigs.get((module, name))
            if params is None:
                log(f"  note: {module}.{name} (legacy {f}) is in the yaml but not in operations/{module}.py")
                continue
            amap = argmap(f, params, hargs)
            lost = [p for p in params if p not in amap and _key(p) not in IGNORE]
            if lost:
                res.setdefault("unpaired_params", []).append((f, name, lost))
            for label, kw, z, ab, pre, sel, exc in (expand(meta) if meta else []):
                hits = []
                for c in calls:
                    extra = match_one(f, c, hargs, kw, (z, ab, pre), params, amap, name, hdef)
                    if extra is not None:
                        hits.append((c, extra))
                if label in all_labels:
                    res.setdefault("duplicate_labels", []).append((label, f, all_labels[label][0]))
                all_labels[label] = (f, module, name)
                if not hits:
                    res["extra"].append((label, f, module, name, kw, z, ab, pre))
                    continue
                matched_cfg.add(label)
                for c, e in hits:
                    pair_for_label.setdefault(label, []).append(c.label)
                    if c.label in matched_calls:
                        res["multi"].append((c.label, label))
                    matched_calls.add(c.label)
                    if e:
                        res["extras_ignored"].append((c.code, label, e))
        # calls with no config
        for c in calls:
            if c.label not in matched_calls:
                res["missing"][f].append(c)
    for module, name, leg, meta in deregistered:
        labs = [lab for lab, *_ in expand(meta)]
        res["deregistered"].append((module, name, leg, labs))
    for module, name in no_legacy:   # no legacy_name: nothing in hctsa to pair with
        res["deregistered"].append((module, name, "(no legacy_name)", [lab for lab, *_ in expand(cfg[module][name])]))
    n_cfg = len(all_labels) + sum(len(v[3]) for v in res["deregistered"])

    # (c) output names
    if run:
        outs = run_outputs(yaml_path, set(pair_for_label), series if series is not None else default_series())
        for label, calls in pair_for_label.items():
            o = outs.get(label)
            if isinstance(o, str):
                res["fields"].append((label, calls[0], "run", o))
                continue
            for cl in calls:
                hf = [f for f, _ in ops.get(cl, []) if f != ""]
                has_scalar = any(f == "" for f, _ in ops.get(cl, []))
                if o is None and hf:
                    res["fields"].append((label, cl, "scalar", f"hctsa has fields {hf[:6]}"))
                elif o is not None and has_scalar and not hf:
                    res["fields"].append((label, cl, "dict", f"hctsa has one output; pyhctsa returns {o[:6]}"))
                elif o is not None:
                    miss, extra = sorted(set(hf) - set(o)), sorted(set(o) - set(hf))
                    res.setdefault("outputs", {})[label] = {"hctsa": sorted(hf), "pyhctsa": o}
                    if miss or extra:
                        res["fields"].append((label, cl, "fields", {"hctsa_only": miss, "pyhctsa_only": extra}))
    res["counts"] = {
        "hctsa_calls": len(mops), "hctsa_features": sum(len(v) for v in ops.values()),
        "yaml_configs": n_cfg, "calls_matched": len(matched_calls),
        "hctsa_calls_implemented": sum(len(calls_of[f]) for f in list(impl) + list(unregistered)),
        "missing_calls": sum(len(v) for v in res["missing"].values()), "extra_configs": len(res["extra"]) +
        sum(len(v[3]) for v in res["deregistered"]),
    }
    res["matched_labels"] = sorted(matched_cfg)
    res["pairs"] = pair_for_label
    res["_ops"], res["_calls_of"], res["_impl"] = ops, calls_of, impl
    return res


# ---------------------------------------------------------------------------------------------------------------
# reporting and yaml suggestions
# ---------------------------------------------------------------------------------------------------------------
def _yaml_value(v) -> str:
    if isinstance(v, float) and not math.isfinite(v):
        return ".nan" if math.isnan(v) else (".inf" if v > 0 else "-.inf")
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v) if "'" not in v else '"' + v + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_yaml_value(x) for x in v) + "]"
    return repr(v)


def suggest_entries(hctsa: Path, res: dict, sigs: dict) -> list[dict]:
    """The yaml to add, per implemented function with missing calls: ``{func, module, name, new, ordered_args, lines}``
    where each line is ``(config, hctsa code)`` (mapped arguments only; review them). ``new`` is True for a function
    that has no yaml entry yet (the whole entry must be added)."""
    impl_by_func = {f: [(m, n) for m, n, _ in v] for f, v in res["_impl"].items()}
    new = set(res["unregistered"])
    for f, (m, n) in res["unregistered"].items():
        impl_by_func.setdefault(f, []).append((m, n))
    out = []
    for f, calls in sorted(res["missing"].items()):
        if f not in impl_by_func:
            continue
        hargs = hctsa_signature(hctsa, f) or []
        groups = defaultdict(list)   # python function that takes the call -> calls
        for c in calls:
            given = given_args(c, hargs)
            for module, name in impl_by_func[f]:
                req = REQUIRE.get((f, name))
                if req and norm(req[2] if given.get(req[0]) is None else given[req[0]]) != norm(req[1]):
                    continue
                groups[(module, name)].append(c)
                break
        for (module, name), gcalls in groups.items():
            params = sigs[(module, name)]
            amap = argmap(f, params, hargs)
            rows = []   # per call: (values by python param, input flags, code)
            for c in gcalls:
                given = given_args(c, hargs)
                vals = {}
                for p in params:
                    if p not in amap or _key(p) in IGNORE:
                        continue
                    a, k = amap[p] if isinstance(amap[p], tuple) else (amap[p], None)
                    v = given.get(a)
                    if k is not None:
                        v = v[k] if isinstance(v, list) and len(v) > k else None
                    if v is None:
                        continue
                    v = python_name(v)
                    if p in PARAM_FORMS:
                        v = PARAM_FORMS[p][1](v)
                    vals[p] = [v] if isinstance(v, list) else v   # a list-valued setting is a nested list in the yaml
                rows.append((vals, INPUTS.get(c.input.replace(" ", ""), (True, False, None)), c.code))
            seen = defaultdict(list)
            for vals, _, _ in rows:
                for p, v in vals.items():
                    seen[p].append(json.dumps(v))
            # parameters that tell the calls apart (they go into the label); the others only when off their default
            ordered = [p for p in params if p in seen and (len(set(seen[p])) > 1 or len(seen[p]) < len(rows))]
            lines = []
            for vals, (z, ab, pre), code in rows:
                kv = [f"{p}: {_yaml_value(v)}" for p, v in vals.items()
                      if p in ordered or params[p] is None or norm(v) != norm(params[p])]
                kv.append(f"zscore: {z}")
                if ab:
                    kv.append("abs: True")
                if pre:
                    kv.append(f"preprocess: {pre}")
                lines.append(("{" + ", ".join(kv) + "}", code))
            out.append({"func": f, "module": module, "name": name, "new": f in new, "ordered_args": ordered,
                        "lines": lines, "unpaired": [p for p in params if p not in amap]})
    return out


def suggest_field_filters(res: dict) -> list[str]:
    """Lines naming, per config label, the outputs pyhctsa returns that hctsa does not register (needs ``--run``): add
    ``exclude: [...]`` to the config (or ``select: [...]`` with hctsa's fields when that is the shorter list)."""
    out = []
    for label, o in sorted(res.get("outputs", {}).items()):
        if o["pyhctsa"] is None:
            continue
        extra = sorted({k.split(".")[0] for k in set(o["pyhctsa"]) - set(o["hctsa"])})
        if extra:
            keep = sorted({k.split(".")[0] for k in o["hctsa"]})
            out.append(f"# {label}: " + (f"select: {keep}" if len(keep) < len(extra) else f"exclude: {extra}"))
    return out


def format_suggestions(entries: list[dict]) -> str:
    out = []
    for e in entries:
        out.append(f"# {e['module']}.{e['name']}  (legacy {e['func']})" + ("  NEW FUNCTION" if e["new"] else "")
                   + f"   ordered_args: {e['ordered_args']}   unpaired python params: {e['unpaired']}")
        out += [f"      - {cfg}   # {code}" for cfg, code in e["lines"]]
    return "\n".join(out)


def report(res: dict, verbose: bool, log=print) -> int:
    n_bad = 0
    cn = res["counts"]
    log(f"hctsa: {cn['hctsa_calls']} calls, {cn['hctsa_features']} features; "
        f"pyhctsa: {cn['yaml_configs']} configs")
    log(f"  hctsa calls of implemented functions: {cn['hctsa_calls_implemented']}, matched by a config: "
        f"{cn['calls_matched']}, missing: {cn['missing_calls']}; configs with no hctsa call: {cn['extra_configs']}")
    log("\n(a) hctsa calls of implemented functions with no matching config")
    for f, calls in sorted(res["missing"].items()):
        if f in res["unregistered"] or f in res["_impl"]:
            log(f"  {f}: {len(calls)}" + ("  [function not in the yaml]" if f in res["unregistered"] else ""))
            for c in calls[: (None if verbose else 4)]:
                log(f"      {c.code}   ({c.label})")
            if not verbose and len(calls) > 4:
                log(f"      ... {len(calls) - 4} more")
            n_bad += len(calls)
    log("\n(b) pyhctsa configs with no hctsa counterpart")
    for module, name, leg, labs in res["deregistered"]:
        log(f"  {module}.{name}: legacy function {leg} is not in hctsa's master operations ({len(labs)} configs)")
        n_bad += len(labs)
    byfn = defaultdict(list)
    for label, f, module, name, kw, z, ab, pre in res["extra"]:
        byfn[(module, name, f)].append((label, kw, z, ab, pre))
    for (module, name, f), items in sorted(byfn.items()):
        log(f"  {module}.{name} ({f}): {len(items)} configs match no hctsa call")
        for label, kw, z, ab, pre in items[: (None if verbose else 6)]:
            log(f"      {label}  {kw}{' zscore' if z else ''}{' abs' if ab else ''}{' ' + pre if pre else ''}")
        if not verbose and len(items) > 6:
            log(f"      ... {len(items) - 6} more")
        n_bad += len(items)
    if res["fields"]:
        log("\n(c) output-name mismatches (hctsa fields against the keys pyhctsa returns)")
        for label, cl, kind, detail in res["fields"]:
            log(f"  {label} <-> {cl}: {kind}: {detail}")
        n_bad += len(res["fields"])
    log("\n(d) hctsa functions with no pyhctsa implementation")
    for f in res["not_implemented"]:
        log(f"  {f}: {len(res['_calls_of'][f])} calls")
    if res.get("unpaired_params"):
        log("\n(d) python parameters with no hctsa counterpart (must stay at their defaults; extend ARGMAP if they pair up)")
        for f, name, lost in res["unpaired_params"]:
            log(f"  {f} ~ {name}: {lost}")
    if res.get("duplicate_labels"):
        log("\n(e) feature labels produced by more than one config (the later one silently replaces the earlier)")
        for label, f, f0 in res["duplicate_labels"]:
            log(f"  {label} ({f}; first seen under {f0})")
        n_bad += len(res["duplicate_labels"])
    if res["extras_ignored"]:
        log("\n(d) calls matched only by ignoring explicitly set hctsa arguments pyhctsa lacks")
        for code, label, extra in res["extras_ignored"][: (None if verbose else 20)]:
            log(f"  {code}  ~ {label}: ignored {extra}")
    if res["multi"]:
        log(f"\n(d) {len(res['multi'])} hctsa calls matched by more than one config, e.g. {res['multi'][:3]}")
    if res["unregistered"]:
        log("\nimplemented in pyhctsa but not in the yaml (matched by name): "
            + ", ".join(f"{f}={m}.{n}" for f, (m, n) in sorted(res["unregistered"].items())))
    return n_bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hctsa", type=Path, help="a hctsa checkout")
    ap.add_argument("--yaml", type=Path, default=REPO / "pyhctsa" / "configurations" / "hctsa.yaml")
    ap.add_argument("--run", action="store_true", help="run each matched config on a test series and compare output names")
    ap.add_argument("--series", type=Path, help="test series for --run (.npy or text); default: a generated AR(1) + sine")
    ap.add_argument("--suggest", action="store_true", help="print yaml for the missing calls of implemented functions")
    ap.add_argument("--json", type=Path, help="write the findings to this file")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every item")
    a = ap.parse_args(argv)
    series = None
    if a.series:
        import numpy as np
        series = [np.load(a.series) if a.series.suffix == ".npy" else np.loadtxt(a.series)]
    res = compare(a.hctsa, a.yaml, run=a.run, series=series)
    n_bad = report(res, a.verbose)
    if a.suggest:
        print("\n# ---- suggested configs ----")
        print(format_suggestions(suggest_entries(a.hctsa, res, py_signatures(REPO / "pyhctsa" / "operations"))))
        if a.run:
            print("\n# ---- outputs hctsa does not register ----")
            print("\n".join(suggest_field_filters(res)))
    if a.json:
        def ser(o):
            if isinstance(o, Call):
                return {"code": o.code, "label": o.label}
            if isinstance(o, (set, tuple)):
                return list(o)
            return str(o)
        keep = {k: v for k, v in res.items() if not k.startswith("_")}
        a.json.write_text(json.dumps(keep, indent=1, default=ser))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
