#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE cache-miss forecasting instance to stdout.

Setting (hidden from the solver -- only the SHAPE is real, not the numbers):
A workload with working-set size N is run against an LRU-managed cache with
capacity C lines and A-way associativity.  Every access that reuses an item
has a "reuse distance" (how many other distinct items were touched since the
item's previous access).  In this workload family the reuse distance of a
repeated access scales with the working-set size N -- a bigger footprint
pushes reuses farther apart in the LRU stack -- so whether an access is a
cache hit or miss depends on N only through the ratio of N to an EFFECTIVE
capacity K(C,A).  K(C,A) is a PUBLIC structural fact (stated to the solver):
associativity A recovers a fraction A/(A+1) of the raw capacity C from
conflict misses, so K = C * A / (A+1).  What is NOT public is exactly how
sharply the miss rate transitions from "everything fits" to "everything
misses" as N crosses K -- that shape is a property of the workload's
locality character, and it is only discoverable from a reuse-distance
sample, never from the small-N miss-rate log alone (because at small N the
transition hasn't happened yet, so the log looks almost flat regardless of
the shape).

STDOUT prints, all whitespace-separated, in order:
  t C A M n_train
  d_1 ... d_M                      (M normalized reuse-distance samples)
  N_1 missrate_1                   (n_train small-working-set measurements)
  ...
  N_{n_train} missrate_{n_train}

The hidden shape parameter, the seed, and the effective-capacity crossing
point are NEVER printed -- only measured data.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
C_LO, C_HI = 800, 30000
A_CHOICES = [1, 2, 4, 8, 16, 32]
S_LO, S_HI = 1.3, 4.6
M_HIST = 60
N_TRAIN = 14
TRAIN_FRAC_LO, TRAIN_FRAC_HI = 0.01, 0.15     # N_i = K * frac, deep sub-cliff
TTRIAL_TRAIN = 6000                            # accesses sampled per train row


def params(t):
    """Hidden instance parameters (identical in gen.py and verify.py)."""
    rng = random.Random(3010001 + t * 800213)
    C = rng.randint(C_LO, C_HI)
    A = rng.choice(A_CHOICES)
    s = rng.uniform(S_LO, S_HI)               # HIDDEN shape (steepness)
    K = C * A / (A + 1.0)                      # PUBLIC effective capacity
    return C, A, s, K


def true_missrate(N, K, s):
    """P(reuse distance > K) under the (hidden) log-logistic reuse-distance
    law scaled by N: a smooth sigmoid in log(N) crossing 0.5 exactly at N=K,
    with steepness s.  Never revealed to the solver in closed form."""
    r = (N / K) ** s
    return r / (1.0 + r)


def gen_histogram(t, s):
    """M normalized reuse-distance samples X ~ log-logistic(shape=s, scale=1)
    via inverse CDF; these are what "measurable in the small-size regime"
    means -- a distributional sample, independent of any particular N."""
    rng = random.Random(5170003 + t * 91711)
    xs = []
    for _ in range(M_HIST):
        u = 0.001 + 0.998 * rng.random()
        x = (u / (1.0 - u)) ** (1.0 / s)
        xs.append(x)
    return xs


def gen_train(t, K, s):
    """n_train (N_i, measured missrate_i) rows, all deep in the sub-cliff
    regime (N_i << K), measured as a finite-sample binomial rate -- so at
    small N the log can look exactly flat/zero even though the true curve
    is not."""
    rng = random.Random(2290007 + t * 60013)
    rows = []
    for _ in range(N_TRAIN):
        frac = math.exp(rng.uniform(math.log(TRAIN_FRAC_LO), math.log(TRAIN_FRAC_HI)))
        N = max(1.0, K * frac)
        p = true_missrate(N, K, s)
        misses = 0
        for _ in range(TTRIAL_TRAIN):
            if rng.random() < p:
                misses += 1
        rows.append((N, misses / float(TTRIAL_TRAIN)))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    C, A, s, K = params(t)
    xs = gen_histogram(t, s)
    train = gen_train(t, K, s)

    out = ["%d %d %d %d %d" % (t, C, A, M_HIST, N_TRAIN)]
    out.append(" ".join("%.6f" % x for x in xs))
    for N, mr in train:
        out.append("%.6f %.6f" % (N, mr))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
