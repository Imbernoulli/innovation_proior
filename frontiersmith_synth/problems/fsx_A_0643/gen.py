#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

A towed-sphere flume logs the drag coefficient Cd of a fixed-size sphere across
a range of Reynolds numbers Re = (density * velocity * diameter) / viscosity
(the rig fixes density, diameter and viscosity per run and sweeps velocity, so
each row is effectively one (density, velocity, viscosity) configuration
collapsed to its Re).  The TRUE law behind Cd(Re) is a hidden TWO-TERM
crossover:

    Cd(Re) = A * Re^p  +  B * Re^q          (p < q < 0, both hidden)

The first term (steep exponent p, near classical Stokes drag) dominates
completely at the LOW-TO-MODERATE Reynolds numbers the flume can reach; the
second term (shallow exponent q) decays far more slowly and, unseen by the
flume, comes to DOMINATE at high Re.  Each testId fixes a DIFFERENT hidden
sphere/fluid combination (different p, A, q, B).

The solver only ever SEES this TRAIN sample: Re values log-uniformly swept
across the flume's reachable range, each with a small multiplicative
measurement-noise floor.  The held-out grading grid (deep in the high-Re
regime the flume cannot reach) is regenerated only inside the checker -- it is
NEVER printed here, and neither are p, A, q, B or any seed.

STDOUT prints ONLY: a header "<testId> <N>" then N rows "Re Cd", one
configuration per line.
"""
import sys, random, math

RE_LO = 0.5
RE_HI = 25.0
N_TRAIN = 90
NOISE_SIGMA = 0.020        # small multiplicative log-noise (measurement floor)


def hidden_law(t):
    """Hidden crossover law for this test id. Lives in gen AND grader; never printed."""
    rng = random.Random(643000 + t * 7919)
    p = -1.0 + rng.uniform(-0.08, 0.08)          # dominant exponent, Stokes-ish
    A = rng.uniform(18.0, 30.0)                  # dominant coefficient
    plan_q = {1: -0.12, 2: -0.18, 3: -0.35, 4: -0.40, 5: -0.15,
              6: -0.10, 7: -0.45, 8: -0.20, 9: -0.30, 10: -0.08}
    q = plan_q.get(t, -0.20) + rng.uniform(-0.02, 0.02)   # subdominant exponent
    B = rng.uniform(0.15, 0.35)                  # subdominant coefficient
    return p, A, q, B


def Cd_true(Re, p, A, q, B):
    return A * (Re ** p) + B * (Re ** q)


def train_rows(t):
    p, A, q, B = hidden_law(t)
    rng = random.Random(90210 + t * 13)
    log_lo, log_hi = math.log(RE_LO), math.log(RE_HI)
    rows = []
    for i in range(N_TRAIN):
        # log-uniform sweep, jittered so points aren't a perfect deterministic grid
        frac = (i + rng.uniform(0.05, 0.95)) / N_TRAIN
        frac = min(0.999999, max(0.000001, frac))
        Re = math.exp(log_lo + frac * (log_hi - log_lo))
        clean = Cd_true(Re, p, A, q, B)
        noisy = clean * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        rows.append((Re, noisy))
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = train_rows(t)
    out = ["%d %d" % (t, len(rows))]
    for Re, Cd in rows:
        out.append("%.8f %.8f" % (Re, Cd))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
