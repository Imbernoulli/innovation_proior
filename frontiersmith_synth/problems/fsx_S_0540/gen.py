#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE alien-billiards logbook (training sample) to stdout.

Theme: alien billiards lab logbooks.  In the lab's basement two fixed alien
billiard balls (unknown, fixed masses) are fired head-on at each other over and
over.  Each head-on collision is a fixed DETERMINISTIC map: the two pre-collision
velocities (u1, u2) turn into two post-collision velocities (v1, v2).  The map
EXACTLY conserves two bookkeeping quantities of the alien physics -- a
momentum-like invariant and an energy-like invariant -- but the *constants* of
those invariants (the two masses and the lab's peculiar velocity ceiling) are
never written in the logbook.

Crucially the logbook only ever records GENTLE (low-energy) shots: |velocity|
stays well inside the ceiling, where the map looks almost like a plain linear
exchange.  The grader later asks you to predict v1 for VIOLENT (high-energy)
shots near the ceiling -- an extrapolation regime that is regenerated only
inside the grader and is never printed here.

Difficulty ladder (testId 1..10): more measurement noise + fewer logged shots.
STDOUT prints ONLY: a header "<n_shots> <test_id>" then n_shots rows
    u1 u2 v1 v2
The masses, the velocity ceiling, and the seed are NOT printed.
"""
import sys
import math
import random


def params(t):
    """Hidden per-test constants: mass ratio r=m2/m1 (m1:=1) and velocity
    ceiling c.  r is drawn clearly away from 1 (equal masses would degenerate
    the map to a pure velocity swap and hide the ceiling)."""
    rng = random.Random(770001 + t * 5303)
    if t % 2 == 0:
        r = rng.uniform(1.70, 2.90)
    else:
        r = rng.uniform(0.30, 0.62)
    c = rng.uniform(1.60, 2.60)
    return r, c


def gamma(u, c):
    return 1.0 / math.sqrt(abs(1.0 - (u / c) ** 2))


def post(u1, u2, r, c):
    """Post-collision velocities of the two-ball elastic collision under the
    lab's ceiling-c physics (m1=1, m2=r).  Conserves p = sum g*m*u and
    E = sum g*m; the physical (exchange) branch is taken."""
    g1 = gamma(u1, c)
    g2 = gamma(u2, c)
    p = g1 * u1 + g2 * r * u2
    E = g1 + g2 * r
    V = p / E
    c2 = c * c
    w1 = (u1 - V) / (1.0 - u1 * V / c2)
    v1 = (V - w1) / (1.0 - w1 * V / c2)
    w2 = (u2 - V) / (1.0 - u2 * V / c2)
    v2 = (V - w2) / (1.0 - w2 * V / c2)
    return v1, v2


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    r, c = params(t)
    sigma = 0.008 + (t - 1) * 0.004
    n = 190 - (t - 1) * 12
    rng = random.Random(4200 + t * 99131)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        # gentle / low-energy shots: speed fraction |s| <= 0.55 of the ceiling
        s1 = rng.uniform(-0.55, 0.55)
        s2 = rng.uniform(-0.55, 0.55)
        u1 = c * s1
        u2 = c * s2
        v1, v2 = post(u1, u2, r, c)
        v1 += rng.gauss(0.0, sigma)
        v2 += rng.gauss(0.0, sigma)
        out.append("%r %r %r %r" % (u1, u2, v1, v2))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
