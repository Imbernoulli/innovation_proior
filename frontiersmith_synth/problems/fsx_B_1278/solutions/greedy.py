# TIER: greedy
"""
The "obvious" textbook fix once you know naive repeated testing is unsafe:
a classic Pocock-style group-sequential plan -- K_max looks equally spaced
from patient 1 to N_max, ONE flat z-threshold shared by every look, no
futility rule, no use of the effect-onset information in the instance. The
flat threshold is calibrated (by bisection over the SAME deterministic null
ensemble the checker uses) to spend ~90% of the alpha budget in total, so it
always respects the type-I cap. It ignores `onset_frac` completely: on
instances where the treatment effect only manifests late, most of its early
looks carry almost no power, and spending equal boundary stringency on them
starves the power available once the effect has actually appeared --
dropping overall power below the required floor.
"""
import sys
import math
import numpy as np

M = 4000


def calibrate_single(cumsum, n_list, target_alpha):
    Mrows = cumsum.shape[0]
    zmax = np.full(Mrows, -1e9)
    for n_i in n_list:
        Z = cumsum[:, n_i - 1] / math.sqrt(n_i)
        zmax = np.maximum(zmax, Z)
    zs = np.sort(zmax)[::-1]
    k = max(1, int(round(target_alpha * Mrows)))
    k = min(k, Mrows - 1)
    return float(zs[k - 1])


def main():
    toks = sys.stdin.read().split()
    N_max = int(toks[0]); K_max = int(toks[1])
    alpha_total = float(toks[2])
    rng_seed = int(toks[7])

    K = K_max
    n_list = [max(1, round(i * N_max / K)) for i in range(1, K + 1)]
    for i in range(1, len(n_list)):
        if n_list[i] <= n_list[i - 1]:
            n_list[i] = n_list[i - 1] + 1
    n_list[-1] = N_max

    rng_null = np.random.default_rng(rng_seed)
    null_sig = rng_null.standard_normal((M, N_max))
    cumsum_null = np.cumsum(null_sig, axis=1)

    target_alpha = 0.90 * alpha_total
    thr = calibrate_single(cumsum_null, n_list, target_alpha)

    parts = [str(K)]
    for n_i in n_list:
        parts += [str(n_i), "%.6f" % thr, "-50.000000"]
    print(" ".join(parts))


if __name__ == "__main__":
    main()
