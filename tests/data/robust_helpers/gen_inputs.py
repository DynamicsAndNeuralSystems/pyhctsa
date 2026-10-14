import json, numpy as np
rng = np.random.default_rng(20261006)
J = {}
def ar1(n, phi):
    e = rng.standard_normal(n); x = np.zeros(n)
    for i in range(1, n): x[i] = phi*x[i-1] + e[i]
    return x
# generic series
J['ar_pos'] = ar1(300, 0.8)
J['ar_neg'] = ar1(300, -0.7)
J['white'] = rng.standard_normal(250)
J['trend'] = np.linspace(0, 5, 200) + 0.3*rng.standard_normal(200)
J['quant'] = np.round(ar1(400, 0.9)*1.5)        # many ties
J['binary'] = (ar1(300, 0.5) > 0).astype(float)
J['mostmax'] = np.concatenate([np.full(60, 3.0), rng.standard_normal(30)])
J['const'] = np.full(50, 2.0)
J['two'] = np.array([1.0, 2.0])
J['small'] = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
J['lattice'] = np.arange(0, 100, 1.0)[rng.permutation(100)]/10   # edges exactly on lattice
J['skew'] = np.exp(rng.standard_normal(500))
J['bimodal'] = np.concatenate([rng.standard_normal(300), 4 + 0.7*rng.standard_normal(200)])
J['nan_series'] = np.concatenate([rng.standard_normal(20), [np.nan], rng.standard_normal(20)])
J['hsm_even'] = np.array([1.0, 2, 2, 3, 7, 8, 9, 9.5])
J['hsm_tie'] = np.array([1.0, 2, 3])
J['hsm_tie2'] = np.array([1.0, 2, 4])
J['hsm_5'] = np.array([0.0, 1, 1.2, 1.3, 9])
# regression data
x = np.linspace(0, 3, 60)
J['x'] = x
J['y_lin_out'] = 2*x + 1 + 0.1*rng.standard_normal(60); J['y_lin_out'][[5, 30, 31]] += [8, -6, 9]
J['y_decay'] = 3*np.exp(-1.5*x) + 0.5 + 0.05*rng.standard_normal(60)
J['y_grow'] = 0.2*np.exp(0.9*x) + 0.02*rng.standard_normal(60)
J['y_nearlin'] = 0.5*x + 0.01*rng.standard_normal(60)
J['y_step'] = (x > 1.5).astype(float) + 0.01*rng.standard_normal(60)
J['x_ties'] = np.repeat(np.arange(10.0), 3)
J['y_ties'] = np.repeat(np.arange(10.0), 3)**1.5 + rng.standard_normal(30)
# residual stats
J['res'] = ar1(120, 0.6)
J['res_exact'] = 1e-9*rng.standard_normal(120)
# densities
xc = np.linspace(-4, 4, 41)
J['xc'] = xc
J['p_gauss'] = np.exp(-(xc-0.3)**2/(2*1.1**2))/ (1.1*np.sqrt(2*np.pi)) + 0.002*rng.random(41)
J['p_bimod'] = 0.6*np.exp(-(xc+1.5)**2/(2*0.6**2))/(0.6*np.sqrt(2*np.pi)) + 0.4*np.exp(-(xc-1.8)**2/(2*0.8**2))/(0.8*np.sqrt(2*np.pi)) + 0.003*rng.random(41)
J['xp'] = np.linspace(0.5, 12, 24)
J['p_power'] = 0.8*J['xp']**-1.6 * (1 + 0.1*rng.standard_normal(24))
J['p_exp'] = np.exp(-0.5*np.linspace(0, 6, 30))*(1+0.05*rng.standard_normal(30)); J['xe'] = np.linspace(0, 6, 30)
# sinusoid series
t = np.arange(1, 201)
J['sin2'] = 1.5*np.sin(2*np.pi*0.071*t + 0.4) + 0.8*np.cos(2*np.pi*0.23*t) + 0.3*rng.standard_normal(200)
J['sin_noise'] = rng.standard_normal(150)
json.dump({k: np.asarray(v).tolist() for k, v in J.items()}, open('inputs.json', 'w'))
