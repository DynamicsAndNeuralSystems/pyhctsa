"""Builds tests/data/symbolic_tau1.json: outputs of the symbolic word and transition features at
tau = 1 (motif_two, motif_three, transition_matrix, transition_p_alphabet), for
tests/test_rank_tau.py::test_tau1_values_pinned.

The values were computed with pyhctsa at commit bcb14a8, before words and transitions were
formed from lagged symbols (at tau = 1 the two definitions coincide)::

    python tests/data/symbolic_tau1/build_fixture.py tests/data/symbolic_tau1.json
"""
import json
import sys

import numpy as np

from pyhctsa.operations.symbolic import motif_three, motif_two, transition_matrix, transition_p_alphabet

CALLS = {  # label -> (function, keyword arguments)
    **{f'motif_two_{b}': (motif_two, dict(binarize_how=b, tau=1)) for b in ('diff', 'mean', 'median', 'iqr')},
    **{f'motif_three_{c}': (motif_three, dict(cg_how=c, tau=1)) for c in ('quantile', 'diffquant')},
    **{f'transition_matrix_quantile_{g}': (transition_matrix, dict(how_to_cg='quantile', num_groups=g, tau=1))
       for g in (2, 3, 4, 5)},
    'transition_matrix_diff_3': (transition_matrix, dict(how_to_cg='diff', num_groups=3, tau=1)),
    'transition_matrix_updown_2': (transition_matrix, dict(how_to_cg='updown', num_groups=2, tau=1)),
    'transition_p_alphabet_2_to_40': (transition_p_alphabet, dict(num_groups=list(range(2, 41)), tau=1)),
}


def series() -> dict:
    rng = np.random.default_rng(7)
    e = rng.standard_normal(700)
    ar = np.zeros(700)
    for t in range(1, 700):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    t = np.arange(400)
    return {
        'ar1': ar[200:],
        'ar1_ties': np.round(ar[200:]),
        'sine_noise': np.sin(2 * np.pi * t / 23) + 0.3 * rng.standard_normal(400),
        'iid': rng.standard_normal(300),
        'short': rng.standard_normal(30),
    }


def _float(v):
    v = float(v)
    return None if np.isnan(v) else v


if __name__ == '__main__':
    out = {'series': {k: v.tolist() for k, v in series().items()}, 'values': {}}
    for label, (fn, kw) in CALLS.items():
        out['values'][label] = {}
        for name, y in series().items():
            r = fn(y, **kw)
            out['values'][label][name] = ({k: _float(v) for k, v in r.items()} if isinstance(r, dict)
                                          else _float(r))
    with open(sys.argv[1], 'w') as f:
        json.dump(out, f)
