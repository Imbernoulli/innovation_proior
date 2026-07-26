#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Hidden proportion law of nested frames (family: proportion-cascade-recurrence).
Every nested-frame DESIGN is a sequence of proportions p_1..p_d (one hidden
2nd-order linear recurrence drives ALL designs sharing this testId):

    (p_{k+1} - p*) = alpha*(p_k - p*) + beta*(p_{k-1} - p*)

i.e. deviations from a fixed point p* evolve by a linear recurrence whose
characteristic roots lambda1, lambda2 (the recurrence's "spectrum") govern
how fast/whether the nest converges toward p* or drifts away from it. Each
design's TENSION score is T = (p_d - p*)^2 -- how far the INNERMOST frame's
proportion strays from the hidden harmonic ratio (lower = more harmonious).

Training designs are SHALLOW (depth 3..6): you see the FULL noisy proportion
trace p_1..p_d plus a noisy tension reading T_obs. The GRADED designs are
DEEP (depth 10..14, regenerated only inside the checker): there you are
given ONLY the first two proportions p1,p2 and a target depth d, and must
predict T from those alone -- a genuine extrapolation past what you observed.

STDOUT prints ONLY: header "<n_rows> <test_id>", then n_rows lines
"<d> <p_1> ... <p_d> <T_obs>". The hidden alpha, beta, p*, lambda1, lambda2
and the RNG seed are NEVER printed here -- only noisy data rows.
"""
import sys, random

# Difficulty ladder over testId: dominant-root magnitude bucket, increasing
# from mild (testId 1-2) to strong (testId 9-10) growth (|lambda1|>1). The
# drift is barely visible over the shallow 3..6 training window (at most a
# few-percent-per-step compounding) but dominates by depth 10..14 -- the trap
# for any model that fits a response surface in d instead of the recurrence.
LAM_BUCKETS = {
    1: (1.05, 1.10), 2: (1.05, 1.10),
    3: (1.10, 1.16), 4: (1.10, 1.16),
    5: (1.17, 1.25), 6: (1.17, 1.25),
    7: (1.26, 1.34), 8: (1.26, 1.34),
    9: (1.35, 1.44), 10: (1.35, 1.44),
}


def derive_law(t):
    """Hidden dynamical system for this test id (lives in gen AND checker,
    never printed). Roots lambda1 (dominant, kept positive), lambda2 (smaller
    magnitude, random sign) are drawn directly, then alpha,beta follow from
    Vieta's formulas -- guarantees real, distinct roots by construction."""
    lo, hi = LAM_BUCKETS.get(t, (0.55, 1.48))
    rng = random.Random("pchidden_law_v1_%d" % t)
    for _ in range(4000):
        lam1 = rng.uniform(lo, hi)
        cap = min(0.50, lam1 - 0.08)
        if cap <= 0.05:
            continue
        lam2_mag = rng.uniform(0.05, cap)
        lam2 = rng.choice([1, -1]) * lam2_mag
        if abs(lam1 - lam2) < 0.08:
            continue
        pstar = rng.uniform(0.45, 0.75)
        alpha = lam1 + lam2
        beta = -lam1 * lam2
        return {"lam1": lam1, "lam2": lam2, "alpha": alpha, "beta": beta, "pstar": pstar}
    raise RuntimeError("law derivation failed for t=%d" % t)


def true_seq(p1, p2, d, alpha, beta, pstar):
    devs = [p1 - pstar, p2 - pstar]
    for _ in range(3, d + 1):
        devs.append(alpha * devs[-1] + beta * devs[-2])
    return [pstar + v for v in devs], devs


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    law = derive_law(t)
    alpha, beta, pstar = law["alpha"], law["beta"], law["pstar"]

    n_rows = 160
    rng = random.Random("pchidden_rows_v1_%d" % t)
    lines = []
    for _ in range(n_rows):
        d = rng.randint(3, 6)
        p1 = min(0.98, max(0.05, pstar + rng.uniform(-0.30, 0.30)))
        p2 = min(0.98, max(0.05, pstar + rng.uniform(-0.30, 0.30)))
        seq, devs = true_seq(p1, p2, d, alpha, beta, pstar)
        T_true = devs[-1] * devs[-1]
        obs = []
        for v in seq:
            sd = 0.05 * max(abs(v - pstar), 0.05)
            obs.append(v + rng.gauss(0.0, sd))
        sdT = 0.05 * max(T_true, 0.02)
        T_obs = max(0.0, T_true + rng.gauss(0.0, sdT))
        row = ("%d " % d) + " ".join("%.6f" % x for x in obs) + (" %.6f" % T_obs)
        lines.append(row)

    out = ["%d %d" % (n_rows, t)] + lines
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
