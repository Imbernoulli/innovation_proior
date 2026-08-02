#!/usr/bin/env python3
"""
Deterministic checker for fsx_B_1278 -- "Onset-Matched Alpha Spending".

Reads the instance (<in>) and the participant's monitoring plan (<out>), then:
  1. validates the plan structurally,
  2. estimates its family-wise type-I error and its power under the
     alternative via a large, deterministically-seeded Monte-Carlo ensemble
     (seed = rng_seed from the instance -- same recipe a solver can replicate),
  3. rejects (Ratio 0.0) on any structural violation, on type-I error above
     the cap, or on power below the floor,
  4. otherwise scores the plan's *expected patient-enrollment cost saved*
     against N_max, normalized by an internal non-adaptive reference plan.

Bit-for-bit deterministic: the only randomness is numpy's default_rng seeded
purely from `rng_seed`, which is read from the (fixed, testId-seeded) input.
"""
import sys
import math
import numpy as np

M = 4000          # Monte-Carlo ensemble size (null and alternative each)
W_ALT = 0.5        # fixed prior weight on the alternative being true
Z_BOUND = 50.0      # sanity bound on submitted z-thresholds


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def simulate(n_list, z_eff, z_fut, cumsum):
    """Walk every simulated cohort (rows of `cumsum`) through the K looks.
    Returns (n_used[M], rejected[M])."""
    K = len(n_list)
    Mrows = cumsum.shape[0]
    active = np.ones(Mrows, dtype=bool)
    decided = np.zeros(Mrows, dtype=bool)
    n_used = np.full(Mrows, n_list[-1], dtype=float)
    rejected = np.zeros(Mrows, dtype=bool)
    for i in range(K):
        n_i = n_list[i]
        Z = cumsum[:, n_i - 1] / math.sqrt(n_i)
        cur = active & ~decided
        eff = cur & (Z >= z_eff[i])
        if i < K - 1:
            fut = cur & (~eff) & (Z <= z_fut[i])
        else:
            fut = np.zeros(Mrows, dtype=bool)
        rejected |= eff
        n_used[eff] = n_i
        n_used[fut] = n_i
        decided |= eff | fut
        if i == K - 1:
            n_used[~decided] = n_i
    return n_used, rejected


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    itoks = open(in_path).read().split()
    if len(itoks) < 8:
        fail("malformed instance")
    N_max = int(itoks[0]); K_max = int(itoks[1])
    alpha_total = float(itoks[2]); power_floor = float(itoks[3])
    cost = float(itoks[4]); delta = float(itoks[5]); onset_frac = float(itoks[6])
    rng_seed = int(itoks[7])

    # ---- parse & validate the participant's plan -------------------------
    raw = open(out_path).read().split()
    if len(raw) < 1:
        fail("empty output")
    try:
        K = int(raw[0])
    except ValueError:
        fail("first token (K) not an integer")
    if not (1 <= K <= K_max):
        fail("K=%r out of range [1,%d]" % (raw[0], K_max))
    need = 1 + 3 * K
    if len(raw) != need:
        fail("expected %d tokens, got %d" % (need, len(raw)))

    n_list, z_eff, z_fut = [], [], []
    for i in range(K):
        ntok, etok, ftok = raw[1 + 3 * i], raw[2 + 3 * i], raw[3 + 3 * i]
        try:
            n_i = int(ntok)
            e_i = float(etok)
            f_i = float(ftok)
        except ValueError:
            fail("non-numeric token in look %d" % (i + 1))
        if not (math.isfinite(e_i) and math.isfinite(f_i)):
            fail("non-finite z-threshold at look %d" % (i + 1))
        if abs(e_i) > Z_BOUND or abs(f_i) > Z_BOUND:
            fail("z-threshold out of sane range at look %d" % (i + 1))
        if f_i > e_i + 1e-12:
            fail("futility boundary above efficacy boundary at look %d" % (i + 1))
        if n_i < 1 or n_i > N_max:
            fail("n_%d=%d out of range [1,%d]" % (i + 1, n_i, N_max))
        if i > 0 and n_i <= n_list[-1]:
            fail("look sizes must be strictly increasing (n_%d=%d <= n_%d=%d)"
                 % (i + 1, n_i, i, n_list[-1]))
        n_list.append(n_i); z_eff.append(e_i); z_fut.append(f_i)

    if n_list[-1] != N_max:
        fail("the final look must enroll exactly N_max patients (n_%d=%d != N_max=%d); "
             "you cannot pre-commit to a smaller trial ceiling, only stop it early"
             % (K, n_list[-1], N_max))

    # ---- deterministic Monte-Carlo ensembles ------------------------------
    rng_null = np.random.default_rng(rng_seed)
    null_sig = rng_null.standard_normal((M, N_max))
    cumsum_null = np.cumsum(null_sig, axis=1)

    rng_alt = np.random.default_rng(rng_seed * 2 + 1)
    j = np.arange(1, N_max + 1)
    if onset_frac <= 1e-9:
        ramp = np.ones(N_max)
    else:
        ramp = np.minimum(1.0, j / (onset_frac * N_max))
    mu_alt = delta * ramp
    alt_sig = rng_alt.standard_normal((M, N_max)) + mu_alt[None, :]
    cumsum_alt = np.cumsum(alt_sig, axis=1)

    def evaluate(nl, ze, zf):
        n_null, rej_null = simulate(nl, ze, zf, cumsum_null)
        n_alt, rej_alt = simulate(nl, ze, zf, cumsum_alt)
        mean_n = W_ALT * float(np.mean(n_alt)) + (1 - W_ALT) * float(np.mean(n_null))
        F = cost * (N_max - mean_n)
        power = float(np.mean(rej_alt))
        alpha_hat = float(np.mean(rej_null))
        return F, power, alpha_hat

    F, power_hat, alpha_hat = evaluate(n_list, z_eff, z_fut)

    cap = alpha_total + max(0.01, 3.0 * math.sqrt(alpha_total * (1 - alpha_total) / M))
    if alpha_hat > cap:
        fail("estimated family-wise type-I error %.4f exceeds cap %.4f" % (alpha_hat, cap))
    if power_hat < power_floor:
        fail("estimated power %.4f below required floor %.4f" % (power_hat, power_floor))

    # ---- internal baseline B: non-adaptive 2-look reference (analytic) ---
    n_triv = [max(1, round(N_max / 2)), N_max]
    a1, a2 = 0.05 * alpha_total, 0.95 * alpha_total
    z_triv = [norm_ppf(1 - a1), norm_ppf(1 - a2)]
    zf_triv = [-Z_BOUND, -Z_BOUND]
    B, _, _ = evaluate(n_triv, z_triv, zf_triv)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    ratio = max(0.0, sc / 1000.0)
    print("F=%.4f B=%.4f power=%.4f alpha_hat=%.4f  Ratio: %.6f" % (F, B, power_hat, alpha_hat, ratio))


if __name__ == "__main__":
    main()
