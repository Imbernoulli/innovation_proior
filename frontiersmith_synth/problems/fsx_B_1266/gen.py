#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of 'Allocating Risk, Not Capital' to stdout.

Deterministic: all randomness is seeded from testId only (see build_regime / draw).

Instance: N capital sleeves. Each sleeve has a CALM-regime return distribution (low,
near-uniform cross-sleeve correlation) and a STRESS-regime return distribution (much
higher volatility AND much higher correlation-to-common-shock for the sleeves that were
already high-volatility in calm times -- correlation-regime-shift). The solver sees a
large sample of calm-regime scenarios and a SMALL sample of stress-regime scenarios (it
must estimate the stress joint structure from this noisy sample, not from the calm data).
It must output capital weights subject to a per-sleeve concentration cap AND a group cap
on the high-calm-volatility "cluster". The checker (verify.py) re-derives the identical
regime parameters from testId and scores the submission on a fresh, larger, HELD-OUT
calm+stress sample the solver never sees.
"""
import sys
import numpy as np

N_LADDER = [4, 5, 5, 6, 6, 7, 7, 8, 8, 8]   # difficulty ladder: small -> large/adversarial
C_CALM_VIS = 2000     # calm scenarios shown to the solver (large: calm mean is estimable)
C_STRESS_VIS = 30     # stress scenarios shown to the solver (deliberately small/noisy)


def build_regime(tid):
    """Pure function of tid -> all per-sleeve regime parameters + constraints.
    MUST stay byte-for-byte identical to the copy in verify.py."""
    N = N_LADDER[tid - 1]
    rng = np.random.default_rng(tid * 7919 + 13)

    vol_hi = 0.045 + 0.0035 * tid
    sigma_calm = rng.uniform(0.015, vol_hi, N)
    # calm mean return carries a small, deterministic "risk premium" tilt toward vol --
    # realistic (riskier sleeves advertise better track records) and is exactly what
    # makes naive return-chasing a trap once the regime shifts.
    mu_calm = 0.004 + 0.22 * sigma_calm

    mult = rng.uniform(2.5, 6.5, N)                 # stress-vol multiplier
    sigma_stress = sigma_calm * mult

    order = np.argsort(sigma_calm)
    rank = np.empty(N)
    rank[order] = np.arange(N)
    rank_frac = rank / max(N - 1, 1)                # 0 = lowest calm-vol, 1 = highest

    mu_stress_factor = 0.02 + 2.6 * rank_frac + rng.uniform(-0.06, 0.06, N)
    mu_stress = -sigma_stress * mu_stress_factor

    rho_calm = rng.uniform(0.05, 0.22, N)
    regime_strength = 0.55 + 0.045 * tid
    jitter = rng.uniform(-0.04, 0.04, N)
    # correlation-regime-shift: stress-regime correlation-to-common-shock grows with
    # calm-volatility rank, so the sleeves that look the most "individually diversifying"
    # in calm times are exactly the ones that crash together in stress.
    rho_stress = np.clip(0.10 + regime_strength * rank_frac + jitter, 0.08, 0.98)

    group_cap = 0.45
    cluster = sorted(np.argsort(-sigma_calm)[:max(1, N // 2)].tolist())
    cluster_set = set(cluster)
    # concentration-limit: cluster sleeves get a TIGHTER individual cap than the rest,
    # so no single sleeve can satisfy the whole group cap alone.
    cap = np.array([rng.uniform(0.15, 0.30) if i in cluster_set else rng.uniform(0.30, 0.70)
                     for i in range(N)])

    return dict(N=N, sigma_calm=sigma_calm, mu_calm=mu_calm, sigma_stress=sigma_stress,
                mu_stress=mu_stress, rho_calm=rho_calm, rho_stress=rho_stress, cap=cap,
                group_cap=group_cap, cluster=cluster)


def draw(rng, N, C, mu, sigma, rho):
    """One-factor Gaussian scenario draw: C scenarios x N sleeves."""
    Fc = rng.standard_normal(C)
    eps = rng.standard_normal((C, N))
    R = mu[None, :] + sigma[None, :] * (rho[None, :] * Fc[:, None]
                                         + np.sqrt(1.0 - rho ** 2)[None, :] * eps)
    return R


def main():
    tid = int(sys.argv[1])
    assert 1 <= tid <= len(N_LADDER), "testId out of range"
    regime = build_regime(tid)
    N = regime['N']

    rng_vis = np.random.default_rng(tid * 104729 + 5)
    calm_vis = draw(rng_vis, N, C_CALM_VIS, regime['mu_calm'], regime['sigma_calm'], regime['rho_calm'])
    stress_vis = draw(rng_vis, N, C_STRESS_VIS, regime['mu_stress'], regime['sigma_stress'], regime['rho_stress'])

    out = []
    out.append(str(tid))
    out.append(str(N))
    out.append(" ".join("%.6f" % v for v in regime['cap']))
    out.append("%.6f" % regime['group_cap'])
    cluster = regime['cluster']
    out.append(str(len(cluster)) + " " + " ".join(str(i) for i in cluster))
    out.append(str(C_CALM_VIS))
    for r in range(C_CALM_VIS):
        out.append(" ".join("%.8f" % v for v in calm_vis[r]))
    out.append(str(C_STRESS_VIS))
    for r in range(C_STRESS_VIS):
        out.append(" ".join("%.8f" % v for v in stress_vis[r]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
