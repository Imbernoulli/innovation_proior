#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for 'Allocating Risk, Not Capital'.

Reads the instance from <in> (which starts with testId), the participant's capital
weights from <out>. Validates feasibility strictly; on any violation prints Ratio: 0.0
and exits 0.

Re-derives the SAME per-sleeve regime (build_regime -- byte-identical copy of gen.py's
function) from testId, then draws a fresh, larger, HELD-OUT calm+stress scenario sample
(different RNG stream than the one gen.py used for the visible sample) that the solver
never saw. Objective:

    F(w) = mean(held-out calm return of w) / CVaR_alpha(held-out stress loss of w)

i.e. return earned in normal times per unit of tail loss realized in a genuine stress
draw, where CVaR_alpha is the mean loss over the worst alpha-fraction of held-out stress
scenarios BY THE SUBMITTED PORTFOLIO'S OWN loss ranking (so which scenarios count as
"tail" depends on how the submitted sleeves co-move under stress, not on any single
sleeve in isolation).

Internal baseline B: a deliberately naive single-sleeve concentration (chase the sleeve
with the single highest calm volatility, i.e. the flashiest sleeve, up to its cap; spill
into the rest of its cluster, then into the rest of the book). Ratio = min(1, SCALE_FRAC
* F / B). Bit-for-bit deterministic: fixed seeds, no wall-time.
"""
import sys
import math
import numpy as np

N_LADDER = [4, 5, 5, 6, 6, 7, 7, 8, 8, 8]
C_CALM_VIS = 150
C_STRESS_VIS = 30
C_CALM_HO = 3000      # held-out calm sample (mean-return leg)
C_STRESS_HO = 300      # held-out stress sample (tail-loss leg)
ALPHA_HO = 0.10         # CVaR tail fraction over the held-out stress sample
SCALE = 230.0           # Ratio = min(1000, SCALE * F / B) / 1000


def fail(msg):
    print("infeasible: %s" % msg, file=sys.stderr)
    print("Ratio: 0.000000")
    sys.exit(0)


def build_regime(tid):
    """Pure function of tid -> all per-sleeve regime parameters + constraints.
    MUST stay byte-for-byte identical to the copy in gen.py."""
    N = N_LADDER[tid - 1]
    rng = np.random.default_rng(tid * 7919 + 13)

    vol_hi = 0.045 + 0.0035 * tid
    sigma_calm = rng.uniform(0.015, vol_hi, N)
    mu_calm = 0.004 + 0.22 * sigma_calm

    mult = rng.uniform(2.5, 6.5, N)
    sigma_stress = sigma_calm * mult

    order = np.argsort(sigma_calm)
    rank = np.empty(N)
    rank[order] = np.arange(N)
    rank_frac = rank / max(N - 1, 1)

    mu_stress_factor = 0.02 + 2.6 * rank_frac + rng.uniform(-0.06, 0.06, N)
    mu_stress = -sigma_stress * mu_stress_factor

    rho_calm = rng.uniform(0.05, 0.22, N)
    regime_strength = 0.55 + 0.045 * tid
    jitter = rng.uniform(-0.04, 0.04, N)
    rho_stress = np.clip(0.10 + regime_strength * rank_frac + jitter, 0.08, 0.98)

    group_cap = 0.45
    cluster = sorted(np.argsort(-sigma_calm)[:max(1, N // 2)].tolist())
    cluster_set = set(cluster)
    cap = np.array([rng.uniform(0.15, 0.30) if i in cluster_set else rng.uniform(0.30, 0.70)
                     for i in range(N)])

    return dict(N=N, sigma_calm=sigma_calm, mu_calm=mu_calm, sigma_stress=sigma_stress,
                mu_stress=mu_stress, rho_calm=rho_calm, rho_stress=rho_stress, cap=cap,
                group_cap=group_cap, cluster=cluster)


def draw(rng, N, C, mu, sigma, rho):
    Fc = rng.standard_normal(C)
    eps = rng.standard_normal((C, N))
    R = mu[None, :] + sigma[None, :] * (rho[None, :] * Fc[:, None]
                                         + np.sqrt(1.0 - rho ** 2)[None, :] * eps)
    return R


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0
    try:
        tid = int(toks[pos]); pos += 1
        N = int(toks[pos]); pos += 1
        cap = np.array([float(toks[pos + i]) for i in range(N)]); pos += N
        group_cap = float(toks[pos]); pos += 1
        K = int(toks[pos]); pos += 1
        cluster = [int(toks[pos + i]) for i in range(K)]; pos += K
    except (IndexError, ValueError) as e:
        fail("cannot parse instance header: %s" % e)
    if not (1 <= tid <= len(N_LADDER)):
        fail("bad testId in instance")
    return tid, N, cap, group_cap, cluster


def read_output(path, N, cap, group_cap, cluster):
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError as e:
        fail("cannot read output: %s" % e)
    if len(toks) != N:
        fail("expected %d weights, got %d" % (N, len(toks)))
    w = np.zeros(N)
    for i, tok in enumerate(toks):
        try:
            v = float(tok)
        except ValueError:
            fail("token %d: not a number: %r" % (i, tok))
        if not math.isfinite(v):
            fail("token %d: non-finite value" % i)
        w[i] = v
    if np.any(w < -1e-6):
        fail("negative weight")
    w = np.maximum(w, 0.0)
    for i in range(N):
        if w[i] > cap[i] + 1e-6:
            fail("sleeve %d weight %.6f exceeds cap %.6f" % (i, w[i], cap[i]))
    s = w.sum()
    if abs(s - 1.0) > 1e-4:
        fail("weights sum to %.6f, must sum to 1" % s)
    cw = sum(w[i] for i in cluster)
    if cw > group_cap + 1e-6:
        fail("cluster weight %.6f exceeds group cap %.6f" % (cw, group_cap))
    return w


def objective(regime, w, calm_ho, stress_ho):
    mean_R = float((calm_ho @ w).mean())
    stress_losses = -(stress_ho @ w)
    M = stress_ho.shape[0]
    k = max(1, round(ALPHA_HO * M))
    tail = np.sort(stress_losses)[-k:]
    cvar = float(tail.mean())
    if mean_R <= 1e-9:
        return 0.0
    return mean_R / max(cvar, 1e-9)


def worst_single_baseline(regime):
    """Internal baseline: chase the single highest-calm-volatility sleeve up to its cap,
    spill into the rest of its cluster (subject to the group cap), then into the rest of
    the book by remaining capacity. Deliberately naive/undiversified."""
    N = regime['N']; cap = regime['cap']; cluster = regime['cluster']; group_cap = regime['group_cap']
    worst = int(np.argmax(regime['sigma_calm']))
    w = np.zeros(N)
    cluster_used = 0.0
    remaining = 1.0
    cluster_set = set(cluster)
    order = [worst] + [i for i in cluster if i != worst] + [i for i in range(N) if i not in cluster_set]
    for i in order:
        avail = cap[i]
        if i in cluster_set:
            avail = min(avail, group_cap - cluster_used)
        take = max(min(avail, remaining), 0.0)
        w[i] = take
        if i in cluster_set:
            cluster_used += take
        remaining -= take
        if remaining <= 1e-12:
            break
    return w


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    tid, N, cap, group_cap, cluster = read_instance(sys.argv[1])
    w = read_output(sys.argv[2], N, cap, group_cap, cluster)

    regime = build_regime(tid)
    rng_ho = np.random.default_rng(tid * 1300021 + 97)
    calm_ho = draw(rng_ho, N, C_CALM_HO, regime['mu_calm'], regime['sigma_calm'], regime['rho_calm'])
    stress_ho = draw(rng_ho, N, C_STRESS_HO, regime['mu_stress'], regime['sigma_stress'], regime['rho_stress'])

    F = objective(regime, w, calm_ho, stress_ho)

    w_base = worst_single_baseline(regime)
    B = objective(regime, w_base, calm_ho, stress_ho)

    if F <= 1e-9:
        sc = 0.0
    else:
        sc = min(1000.0, SCALE * F / max(B, 1e-9))
        sc = max(0.0, sc)
    print("F=%.6f baseline=%.6f" % (F, B), file=sys.stderr)
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
