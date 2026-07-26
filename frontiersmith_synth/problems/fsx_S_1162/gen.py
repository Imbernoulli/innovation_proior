#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Family: annealing-path-hysteresis ("glassblower's cooling schedules decide
the final color").  A hidden first-order transition has TWO stable branches
m_hot(T), m_cold(T) (both affine in T).  A piece of glass always ENTERS the
schedule molten (hot branch).  While cooling, the hot branch loses stability
and the piece JUMPS to the cold branch the instant T drops through the lower
spinodal T_down.  While re-heating, the cold branch loses stability and the
piece jumps back to the hot branch the instant T rises through the UPPER
spinodal T_up (T_up > T_down): a classic hysteresis loop with a bistable
window (T_down, T_up) in which the outcome depends on which branch you
approached from, NOT on the temperature alone.

Every TRAIN protocol here is a MONOTONE COOLING schedule (a piece pulled hot
from the furnace and cooled to some final temperature).  Under monotone
cooling from a guaranteed-hot start, the branch only ever switches once (hot
-> cold, at T_down), so the final color is fully explained by the ENDPOINT
temperature alone -- T_up is never exercised by this data.  path and endpoint
are confounded by design.  The held-out grading protocols (regenerated only
inside the checker) include reheating schedules that visit the bistable
window from BOTH directions, exposing the confound.

STDOUT prints ONLY:
    T_MIN T_MAX N t
    <K> <T_0> <T_1> ... <T_{K-1}> <m_noisy>      (N of these rows)
The hidden branch coefficients, spinodal temperatures and RNG seed are NEVER
printed -- only protocol breakpoints (temperatures, strictly decreasing) and
a noisy final-state measurement.
"""
import sys, random

T_MIN, T_MAX = 250.0, 950.0
T_CENTER = 600.0
SIGMA = 8.0


def true_law(t):
    """Hidden first-order law for this test id (duplicated verbatim in verify.py)."""
    rng = random.Random(900001 + t * 7919)
    T_up = rng.uniform(540.0, 610.0)
    gap = rng.uniform(70.0, 220.0)
    T_down = T_up - gap
    A1 = rng.uniform(55.0, 65.0)
    B1 = rng.uniform(-0.02, 0.02)
    GAP0 = rng.uniform(22.0, 32.0)
    A2 = A1 - GAP0
    B2 = rng.uniform(-0.02, 0.02)
    return A1, B1, A2, B2, T_down, T_up


def branch_val(T, A, B):
    return A + B * (T - T_CENTER)


def simulate(breakpoints, A1, B1, A2, B2, T_down, T_up):
    """Roll a piecewise-linear protocol through the hysteresis law; return m_final.
    breakpoints[0] MUST be > T_up (guaranteed molten start -> branch='hot')."""
    branch = "hot"
    for i in range(len(breakpoints) - 1):
        Ti, Tj = breakpoints[i], breakpoints[i + 1]
        if branch == "hot" and Tj < Ti:
            if Ti > T_down >= Tj:
                branch = "cold"
        elif branch == "cold" and Tj > Ti:
            if Ti < T_up <= Tj:
                branch = "hot"
    Tend = breakpoints[-1]
    return branch_val(Tend, A1, B1) if branch == "hot" else branch_val(Tend, A2, B2)


def make_monotone_protocol(rng):
    T0 = rng.uniform(860.0, 910.0)
    K = rng.randint(2, 5)
    Tend = rng.uniform(T_MIN, T0 - rng.uniform(20.0, 60.0))
    Tend = max(T_MIN, Tend)
    if K == 2:
        seq = [T0, Tend]
    else:
        mids = sorted((rng.uniform(Tend, T0) for _ in range(K - 2)), reverse=True)
        seq = [T0] + list(mids) + [Tend]
    # enforce strictly decreasing (guards against float ties from uniform draws)
    fixed = [seq[0]]
    for v in seq[1:]:
        if v >= fixed[-1]:
            v = fixed[-1] - 1e-3
        fixed.append(v)
    return fixed


def n_train_for(t):
    return int(25 + 45 * (t - 1))


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    A1, B1, A2, B2, T_down, T_up = true_law(t)
    N = n_train_for(t)
    rng = random.Random(31 + t * 104729)

    lines = ["%.3f %.3f %d %d" % (T_MIN, T_MAX, N, t)]
    for _ in range(N):
        proto = make_monotone_protocol(rng)
        true_m = simulate(proto, A1, B1, A2, B2, T_down, T_up)
        noisy_m = true_m + rng.gauss(0.0, SIGMA)
        toks = [str(len(proto))] + ["%.4f" % v for v in proto] + ["%.6f" % noisy_m]
        lines.append(" ".join(toks))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
