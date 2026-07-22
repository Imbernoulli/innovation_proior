#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN census to stdout.

Lynx-hare census with LAGGED crowding.  The hidden law is a seasonally-forced,
DELAYED logistic map for a normalised population density x (fraction of carrying
capacity):

    x[t+1] = x[t] * r[t mod 4] * (1 - x[t - tau] / K)

with a hidden integer delay `tau` (crowding acts on the density `tau` seasons
ago, not the current one), a hidden 4-season growth table `r[0..3]`, and a hidden
carrying capacity `K`.  Each testId fixes a DIFFERENT hidden ecosystem.

The solver only ever SEES this TRAIN census, recorded during a QUIESCENT stretch:
the population sits near its stable seasonal orbit and is nudged only by small
process noise.  In that regime the delay is nearly invisible -- an ordinary
(undelayed) logistic fit tracks the census to the noise floor.  The HELD-OUT
grading census lives in a very different regime (a large population crash that
sends the true system into delay-induced ringing / overshoot) and is regenerated
only inside the grader -- it is NEVER printed here.

STDOUT prints ONLY: a header "<testId> <N> <S> <MAXLAG>" then N census rows, one
density per line.  The season of row i is (i mod S).  The hidden tau, r, K and
the seeds are NOT printed.
"""
import sys, random, math

S = 4
MAXLAG = 6
N_TRAIN = 220
PROC = 0.03          # small multiplicative process noise (quiescent regime)


def params(t):
    """Hidden ecosystem for this test id.  Lives in gen AND grader; never printed."""
    rng = random.Random(6050000 + t * 7919)
    # (tau, R)  -- R is the mean seasonal growth; picked so the delayed system is
    # STABLE but under-damped (perturbations ring at the delay period and decay).
    plan = {1: (1, 1.30), 2: (1, 1.45), 3: (2, 1.42), 4: (2, 1.44), 5: (2, 1.54),
            6: (3, 1.32), 7: (3, 1.37), 8: (2, 1.50), 9: (2, 1.46), 10: (3, 1.35)}
    tau, R = plan[t]
    R += rng.uniform(-0.012, 0.012)
    amp = rng.uniform(0.06, 0.10)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    K = rng.uniform(0.95, 1.10)
    crash = rng.uniform(0.30, 0.42)
    r = [R * (1.0 + amp * math.sin(2.0 * math.pi * s / S + phase)) for s in range(S)]
    return tau, R, K, r, crash


def true_next(x, r, K, tau, t):
    return x[t - 1] * r[t % S] * (1.0 - x[t - 1 - tau] / K)


def train_census(t):
    tau, R, K, r, crash = params(t)
    rng = random.Random(90210 + t * 13)
    x = [0.5 * K] * (tau + 2)
    for _ in range(3000):                 # settle onto the seasonal orbit
        x.append(true_next(x, r, K, tau, len(x)))
    while len(x) % S != 0:                # align so row 0 is at season 0
        x.append(true_next(x, r, K, tau, len(x)))
    start = len(x)
    for _ in range(N_TRAIN):              # observe with small process noise
        nx = true_next(x, r, K, tau, len(x)) * math.exp(rng.gauss(0.0, PROC))
        x.append(nx)
    return x[start:start + N_TRAIN]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    census = train_census(t)
    out = ["%d %d %d %d" % (t, len(census), S, MAXLAG)]
    for v in census:
        out.append("%.6f" % v)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
