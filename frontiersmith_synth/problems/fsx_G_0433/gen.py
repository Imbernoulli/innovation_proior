#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training point cloud to stdout.

Planetary-ellipse orbit recovery.  A small telescope has logged the sky-plane
positions (x, y) of a single planet as it moves along its orbit.  The Sun sits
at the ORIGIN, which is a FOCUS of the elliptical orbit (Kepler's first law), so
every logged point lies on one hidden conic

        r(phi) = p / (1 + e*cos(phi - omega)),   r = sqrt(x^2 + y^2)

with x = r*cos(phi), y = r*sin(phi).  Equivalently every point satisfies a
hidden implicit conic  F(x, y) = 0.  Each test id fixes a DIFFERENT orbit
(semi-latus rectum, eccentricity, orientation).

The telescope was pointed away from the sky during one contiguous stretch of the
orbit, so a whole ARC of true anomaly is MISSING from the training log.  The
grader scores your recovered curve on that WITHHELD ARC -- a genuine
extrapolation region regenerated only inside the grader; it is never printed
here.

Difficulty ladder (testId 1..10): more positional noise + fewer logged points.
STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train rows "x y".
The hidden orbit and its seed are NOT printed.
"""
import sys
import random
import math

TWO_PI = 2.0 * math.pi


def orbit_params(t):
    rng = random.Random(920177 + t * 5273)
    a = rng.uniform(1.6, 3.4)         # semi-major axis
    e = rng.uniform(0.42, 0.56)       # eccentricity (clearly elliptical)
    omega = rng.uniform(0.0, TWO_PI)  # argument of perihelion (orientation)
    p = a * (1.0 - e * e)             # semi-latus rectum
    # withheld arc of polar angle phi (kept away from wrap-around)
    gwid = rng.uniform(1.0, 1.35)
    glo = rng.uniform(0.35, TWO_PI - gwid - 0.35)
    return a, e, omega, p, glo, glo + gwid


def point(phi, pm):
    a, e, omega, p, glo, ghi = pm
    r = p / (1.0 + e * math.cos(phi - omega))
    return r * math.cos(phi), r * math.sin(phi)


def gen_train(t):
    a, e, omega, p, glo, ghi = pm = orbit_params(t)
    sigma = a * (0.008 + (t - 1) * 0.0012)
    n = 170 - (t - 1) * 7
    rng = random.Random(6600 + t * 74521)
    rows = []
    while len(rows) < n:
        phi = rng.uniform(0.0, TWO_PI)
        if glo <= phi <= ghi:        # this arc was never observed
            continue
        x, y = point(phi, pm)
        x += rng.gauss(0.0, sigma)
        y += rng.gauss(0.0, sigma)
        rows.append((x, y))
    return n, rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n, rows = gen_train(t)
    out = ["%d %d" % (n, t)]
    for x, y in rows:
        out.append("%r %r" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
