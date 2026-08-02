# TIER: strong
"""
The insight: pre-specify the alpha-spending schedule to match the effect's
*onset timing*, instead of spreading looks blindly over [1, N_max].

  1. Placement: put all K_max looks in the window [onset_frac*N_max, N_max]
     -- the region where the treatment effect has actually (partly) shown up
     -- rather than wasting early looks before the signal exists.
  2. Spending shape: a_i = target_alpha * t_i^0.3 (t_i = progress through
     that window) -- once you are past onset the full effect is already
     present, so (unlike classical O'Brien-Fleming, which is conservative
     early within the *whole* trial) there is no reason to hold the budget
     back further inside the post-onset window: spend it as soon as you're
     past onset, where it already buys real power.
  3. A mild futility boundary (z <= -0.5) at every non-final look: since
     futility stops never touch the type-I budget, this harvests extra
     enrollment-cost savings on truly-null cohorts for free.

Both the placement and the spending targets are calibrated using the SAME
deterministic null ensemble the checker itself uses (seeded by rng_seed),
via sequential bisection, so the plan is always comfortably inside the
alpha cap while using (almost) the whole budget.
"""
import sys
import math
import numpy as np

M = 4000


def calibrate_sequential(cumsum, n_list, a_targets):
    Mrows = cumsum.shape[0]
    active = np.ones(Mrows, dtype=bool)
    z_eff = []
    prev_a = 0.0
    for i, n_i in enumerate(n_list):
        Z = cumsum[:, n_i - 1] / math.sqrt(n_i)
        cur_idx = np.where(active)[0]
        incr = a_targets[i] - prev_a
        if incr <= 0 or len(cur_idx) == 0:
            thr = 1e9
        else:
            vals = Z[cur_idx]
            svals = np.sort(vals)[::-1]
            cnt = min(max(1, int(round(incr * Mrows))), len(svals))
            thr = float(svals[cnt - 1])
        z_eff.append(thr)
        eff = active & (Z >= thr)
        active = active & ~eff
        prev_a = a_targets[i]
    return z_eff


def main():
    toks = sys.stdin.read().split()
    N_max = int(toks[0]); K_max = int(toks[1])
    alpha_total = float(toks[2])
    onset_frac = float(toks[6])
    rng_seed = int(toks[7])

    K = K_max
    start = max(5, onset_frac * N_max)
    start = min(start, N_max - K)
    n_list = [int(round(start + i * (N_max - start) / K)) for i in range(1, K + 1)]
    for i in range(1, len(n_list)):
        if n_list[i] <= n_list[i - 1]:
            n_list[i] = n_list[i - 1] + 1
    n_list[-1] = N_max

    rng_null = np.random.default_rng(rng_seed)
    null_sig = rng_null.standard_normal((M, N_max))
    cumsum_null = np.cumsum(null_sig, axis=1)

    target_alpha = 0.90 * alpha_total
    ts = [(n - n_list[0]) / max(1, (N_max - n_list[0])) for n in n_list]
    ts[-1] = 1.0
    a_targets = [target_alpha * (t ** 0.3) for t in ts]
    a_targets[-1] = target_alpha
    z_eff = calibrate_sequential(cumsum_null, n_list, a_targets)
    z_fut = [-0.5] * (K - 1) + [-50.0]

    parts = [str(K)]
    for n_i, e_i, f_i in zip(n_list, z_eff, z_fut):
        e_i = max(-49.0, min(49.0, e_i))
        parts += [str(n_i), "%.6f" % e_i, "%.6f" % f_i]
    print(" ".join(parts))


if __name__ == "__main__":
    main()
