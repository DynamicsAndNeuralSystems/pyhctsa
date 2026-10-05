"""Builds tests/data/p2_a.json: MATLAB (hctsa robust/all + robust/finish) values for tests/test_p2_a.py.

Run from SP/fix/p2-a, where calls.py (the MATLAB and Python call expressions), series.json and the
MATLAB results ml_<group>.json (written by genml.py + ml_<group>.m, run with
HCTSA=SP/wt-rb-all SP/fix/mlrun.sh) live, and ml_SDs.json (SD_MakeSurrogates, SD_SurrogateTest and
SD_Surrogates on 200-point series with 20 surrogates, seed 42)::

    python build_fixture.py /path/to/pyhctsa/tests/data/p2_a.json
"""
import json
import re
import sys

from calls import CALLS

SERIES = ['Z1', 'Z6', 'quant_ar', 'lattice300']
KEYS = {  # group -> keys
    'AN': ['an_1_even_10', 'an_1_gaussian_x', 'an_1_std1_10', 'an_ac1e_even_10', 'an_seed3'],
    'HA': None,
    'CO': ['cmin_even', 'cmin_quantiles', 'e2_tau', 'e2d_tau', 'e2_1', 'e2s_0p1', 'e2s_1', 'pac20', 'pac80', 'stick'],
    'IN': ['gami_1', 'gami_40', 'fmin_ac', 'fmin_mi_gaussian', 'fmax_mi_gaussian', 'fmin_mi_hist5', 'fmin_mi_hist10',
           'mami_ac_gaussian', 'mami_ac1e_gaussian'],
    'IS': None,
    'SY': ['de_hist_5_0', 'de_hist_10_0', 'de_hist_10_0p01', 'de_hist_auto_0p01', 'de_hist_sqrt_0', 'de_hist_fd_0', 'de_hist_sturges_0p02',
           'de_hist_50_0', 'de_ks_x_0', 'de_ks_x_0p01', 'de_ks_0p2_0', 'de_ks_0p2_0p05'] + [f'fs{i}' for i in range(9)] + ['tpa_c', 'tpa_def'],
    'NL': ['d2a', 'dvva', 'dvvb', 'pha'],
    'RN': None,
}
out = {'series': {}, 'calls': {}, 'sd': json.load(open('ml_SDs.json'))}
series = json.load(open('series.json'))
for n in SERIES:
    out['series'][n] = series['y'][series['names'].index(n)]
for g, keys in KEYS.items():
    ml = json.load(open(f'ml_{g}.json'))
    for c in CALLS:
        if c['group'] != g or (keys is not None and c['key'] not in keys):
            continue
        exp = {}
        for n in SERIES:
            sn = re.sub(r'\W', '_', n)
            if sn in ml and c['key'] in ml[sn]:
                exp[n] = ml[sn][c['key']]
        if exp:
            out['calls'][c['key']] = {'group': g, 'py': c['py'], 'expected': exp}
json.dump(out, open(sys.argv[1], 'w'))
