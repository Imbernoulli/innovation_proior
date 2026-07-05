#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Aquarium sump-plumbing bench: a hidden per-rig "return-line flow law" couples
four normalised plumbing readings taken while trimming a reef-tank sump loop

    x1 = return-pump throttle setting     x2 = static head (vertical lift)
    x3 = return-pipe bore (inner width)   x4 = fitting-restriction index
                                               (elbows / valves in the run)

to the delivered return flow y (turnover in tank-volumes/hour, normalised).
Pump delivery falls off EXPONENTIALLY as head rises, the bore contributes a
square-law (cross-section) term, throttle and restriction interact, and there
is a mild linear bore trend plus an offset.  Each test id fixes a DIFFERENT
hidden rig (different pump curve / plumbing) with its own exponential head
decay rate -- the number the solver must recover.

The aquarist only ever tunes inside the SAFE operating core, x_i in [0,1], and
every gauge reading carries pump-ripple + counting noise.  The held-out
grading split lives in the OVER-RANGE frontier (higher head / larger bore) and
is regenerated inside the grader only -- it is never printed here.

Difficulty ladder (testId 1..10): more gauge noise + fewer sampled trims.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train data rows.
The hidden law and its seed are NOT printed.
"""
import sys, random, math


def coeffs(t):
    rng = random.Random(770413 + t * 5087)
    a = rng.uniform(1.5, 3.0)     # x1*exp(c*x2)   throttle * head-decay envelope
    c = rng.uniform(0.50, 0.90)   # HIDDEN exponential head-decay rate
    b = rng.uniform(1.5, 3.0)     # x3^2           bore cross-section
    d = rng.uniform(1.0, 2.5)     # x1*x4          throttle * restriction interaction
    e = rng.uniform(0.5, 1.5)     # x3             linear bore trend
    g = rng.uniform(-0.5, 0.5)    # offset
    return a, c, b, d, e, g


def fval(x, cf):
    a, c, b, d, e, g = cf
    return (a * x[0] * math.exp(c * x[1]) + b * x[2] * x[2]
            + d * x[0] * x[3] + e * x[2] + g)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.07 + (t - 1) * 0.05
    n = 400 - (t - 1) * 22
    cf = coeffs(t)
    rng = random.Random(1290 + t * 90173)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        x = [rng.uniform(0.0, 1.0) for _ in range(4)]
        y = fval(x, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r %r %r %r" % (x[0], x[1], x[2], x[3], y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
