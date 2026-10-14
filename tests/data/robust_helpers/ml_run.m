D = pwd;
S = jsondecode(fileread(fullfile(D, 'inputs.json')));
f = fieldnames(S); for i = 1:numel(f), S.(f{i}) = S.(f{i})(:); end
R = struct();
% --- BF_Random
R.rand_ref = BF_Random(1000, [12345 12345 12345 12345 12345 12345]);
R.rand_s0 = BF_Random(8, 0);
R.rand_s7 = BF_Random(5000, 7);
R.rand_s12345 = BF_Random(6, 12345);
R.rand_normal7 = BF_Random(5001, 7, 'normal');
R.rand_normal0 = BF_Random(6, 0, 'normal');
R.rand_perm3 = BF_Random(50, 3, 'perm');
R.rand_perm0_1 = BF_Random(1, 0, 'perm');
R.rand_vec = BF_Random(5, [1 2 3 4 5 6]);
R.rand_empty = BF_Random(0, 0);
% --- BF_TieBreakNoise
R.tb_quant0 = BF_TieBreakNoise(S.quant, 0);
R.tb_quant1 = BF_TieBreakNoise(S.quant, 1);
R.tb_binary = BF_TieBreakNoise(S.binary, 5);
R.tb_white = BF_TieBreakNoise(S.white, 0);
R.tb_const = BF_TieBreakNoise(S.const, 0);
% --- BF_RunsZ
nm = {'ar_pos','ar_neg','white','trend','quant','binary','mostmax','const','two','small','nan_series','hsm_even'};
R.runs = zeros(numel(nm), 1);
for i = 1:numel(nm), R.runs(i) = BF_RunsZ(S.(nm{i})); end
% --- BF_ResidualStats
[a1, a2, rz] = BF_ResidualStats(S.res, sum((S.res - mean(S.res)).^2) * 3);
R.resstats = [a1 a2 rz];
[a1, a2, rz] = BF_ResidualStats(S.res_exact, 1e6);
R.resstats_exact = [a1 a2 rz];
% --- BF_TheilSen
R.ts1 = BF_TheilSen(S.x, S.y_lin_out);
R.ts2 = BF_TheilSen(S.x_ties, S.y_ties);
R.ts3 = BF_TheilSen(ones(5,1), (1:5)');
R.ts4 = BF_TheilSen(S.x, S.y_decay);
% --- BF_ExpFit
cases = {'y_decay', 'y_grow', 'y_nearlin', 'y_step', 'y_lin_out'};
R.exp = zeros(numel(cases) * 4, 6); k = 0;
for i = 1:numel(cases)
    for wo = [true false]
        for mr = [20 5]
            k = k + 1; q = BF_ExpFit(S.x, S.(cases{i}), wo, mr);
            R.exp(k, :) = [q.a q.b q.c q.r2 q.adjr2 q.rmse];
        end
    end
end
q = BF_ExpFit(S.x, ones(60,1)); R.exp_const = [q.a q.b q.c q.r2 q.adjr2 q.rmse];
q = BF_ExpFit([1;2;3], [1;3;2]); R.exp_small = [q.a q.b q.c q.r2 q.adjr2 q.rmse];
% --- BF_GaussMix2
[w, mu, sg] = BF_GaussMix2(S.xc, S.p_bimod, S.xc(2) - S.xc(1)); R.gm_bimod = [w mu sg];
[w, mu, sg] = BF_GaussMix2(S.xc, S.p_gauss, S.xc(2) - S.xc(1)); R.gm_gauss = [w mu sg];
% --- BF_FitDensityCurve
R.fd_gauss_g = BF_FitDensityCurve(S.xc, S.p_gauss, 'gauss');
R.fd_gauss_b = BF_FitDensityCurve(S.xc, S.p_bimod, 'gauss');
R.fd_gauss2_b = BF_FitDensityCurve(S.xc, S.p_bimod, 'gauss2');
R.fd_gauss2_g = BF_FitDensityCurve(S.xc, S.p_gauss, 'gauss2');
R.fd_exp = BF_FitDensityCurve(S.xe, S.p_exp, 'exp');
R.fd_power = BF_FitDensityCurve(S.xp, S.p_power, 'power');
R.fd_exp_g = BF_FitDensityCurve(S.xc, S.p_gauss, 'exp');
% --- BF_FitSinusoids
[yf, fr] = BF_FitSinusoids(S.sin2, 2); R.sin2_fit = yf; R.sin2_f = fr;
[yf, fr] = BF_FitSinusoids(S.sin2, 1); R.sin1_fit = yf; R.sin1_f = fr;
[yf, fr] = BF_FitSinusoids(S.sin_noise, 3); R.sinn_fit = yf; R.sinn_f = fr;
% --- BF_KSDensity
[fd, xi, h] = BF_KSDensity(S.bimodal); R.ks_bimodal = fd; R.ks_bimodal_xi = xi; R.ks_bimodal_h = h;
[fd, xi, h] = BF_KSDensity(S.skew, linspace(0, 6, 25)); R.ks_skew = fd; R.ks_skew_h = h;
[fd, xi, h] = BF_KSDensity(S.quant, [-3 0 2.5], 0.7); R.ks_quant = fd; R.ks_quant_h = h;
[fd, xi, h] = BF_KSDensity(S.mostmax); R.ks_mostmax = fd; R.ks_mostmax_h = h;
[fd, xi, h] = BF_KSDensity(S.const); R.ks_const_h = h; R.ks_const_f1 = fd(1);
[fd, xi, h] = BF_KSDensity(S.nan_series, [0 1]); R.ks_nan = fd; R.ks_nan_h = h;
% --- BF_HistEdges
hn = {'white','quant','lattice','skew','bimodal','binary','const','small','mostmax','nan_series'};
rules = {'auto','sqrt','sturges','fd'};
for i = 1:numel(hn)
    for j = 1:numel(rules)
        R.(sprintf('he_%s_%s', hn{i}, rules{j})) = BF_HistEdges(S.(hn{i}), rules{j});
    end
end
R.he_lattice_10 = BF_HistEdges(S.lattice, 10);
R.he_lattice_lim = BF_HistEdges(S.lattice, 5, [0 10]);
R.he_white_limrule = BF_HistEdges(S.white, 'sqrt', [-5 5]);
% --- BF_QuantileEdges
for i = 1:numel(hn)
    R.(sprintf('qe_%s_10', hn{i})) = BF_QuantileEdges(S.(hn{i}), 10);
end
R.qe_quant_3 = BF_QuantileEdges(S.quant, 3);
R.qe_white_7 = BF_QuantileEdges(S.white, 7);
R.qe_lattice_4 = BF_QuantileEdges(S.lattice, 4);
% --- BF_HalfSampleMode
hm = {'white','quant','lattice','skew','bimodal','binary','const','two','small','mostmax','nan_series','hsm_even','hsm_tie','hsm_tie2','hsm_5'};
R.hsm = zeros(numel(hm), 1);
for i = 1:numel(hm), R.hsm(i) = BF_HalfSampleMode(S.(hm{i})); end
% --- BF_Random 'perm' (argsort of uniforms) and BF_RandomSeed (added with robust/finish)
R.rand_perm7_4000 = BF_Random(4000, 7, 'perm');
R.rand_perm_vec = BF_Random(12, [1 2 3 4 5 6], 'perm');
R.seed_vals = [BF_RandomSeed('default'), BF_RandomSeed([]), BF_RandomSeed(7.6), BF_RandomSeed(2.5), BF_RandomSeed(-3), BF_RandomSeed(5e9+0.4), BF_RandomSeed(0.5), BF_RandomSeed(42)];
save(fullfile(D, 'results.mat'), '-struct', 'R', '-v7');
disp('done');
