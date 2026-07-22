#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Hidden deterministic map on state (x, y): a "boost" step

    u  = TCAP * tanh( p1*x + p2*y + p3*sin(p4*x + p5*y) + p6*x*y )
    c  = (1 + u^2) / (1 - u^2)
    s  = 2*u / (1 - u^2)
    x' = c*x + s*y
    y' = s*x + c*y

for a DIFFERENT random (p1..p6) per test id t (this is the "hidden law").  This
family of steps ALGEBRAICALLY preserves x^2 - y^2 for ANY choice of u(x,y)
(a Lorentz-boost identity: c^2 - s^2 = 1 identically), no matter how wild the
twist function u is -- the map itself can be highly nonlinear and test-id
specific, yet x^2 - y^2 never moves.

The solver only ever SEES transitions sampled from a NARROW RING close to the
unit circle (train regime).  The grader (verify.py) re-derives the SAME hidden
(p1..p6) from t and evaluates candidate invariants on a held-out ring at MUCH
LARGER radius (extrapolation regime) -- never printed here.

On the narrow training ring, x^2 + y^2 is ALMOST constant purely because of
HOW the ring was sampled (r stays in [0.9, 1.1]) -- a value-space coincidence,
not a property of the dynamics.  x^2 - y^2, the actually conserved quantity,
looks like it varies a lot across the training ring (it depends on angle) even
though it is EXACTLY unchanged by every single transition.

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train rows
"<x> <y> <x'> <y'>" (small measurement noise added to x',y').  The hidden
(p1..p6) and the seed are NOT printed -- data rows only.
"""
import sys, random, math

TCAP = 0.6


def hidden_params(t):
    """Hidden law parameters for this test id (lives in gen AND grader, never printed)."""
    rng = random.Random(900011 + t * 7919)
    p1 = rng.uniform(-1.0, 1.0)
    p2 = rng.uniform(-1.0, 1.0)
    p3 = rng.uniform(0.3, 0.8)
    p4 = rng.uniform(0.5, 2.0)
    p5 = rng.uniform(0.5, 2.0)
    p6 = rng.uniform(-0.5, 0.5)
    return (p1, p2, p3, p4, p5, p6)


def step(x, y, params):
    p1, p2, p3, p4, p5, p6 = params
    g = p1 * x + p2 * y + p3 * math.sin(p4 * x + p5 * y) + p6 * (x * y)
    u = TCAP * math.tanh(g)
    c = (1.0 + u * u) / (1.0 - u * u)
    s = 2.0 * u / (1.0 - u * u)
    xp = c * x + s * y
    yp = s * x + c * y
    return xp, yp


def train_ring(t, n):
    """Restricted TRAIN regime: a narrow ring around radius 1."""
    rng = random.Random(31337 + t * 104729)
    params = hidden_params(t)
    sigma = 0.001 + 0.0006 * (t - 1)
    rows = []
    for _ in range(n):
        r = rng.uniform(0.9, 1.1)
        th = rng.uniform(0.0, 2.0 * math.pi)
        x = r * math.cos(th)
        y = r * math.sin(th)
        xp, yp = step(x, y, params)
        xp += rng.gauss(0.0, sigma)
        yp += rng.gauss(0.0, sigma)
        rows.append((x, y, xp, yp))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = 220 - 10 * (t - 1)
    rows = train_ring(t, n)
    out = ["%d %d" % (n, t)]
    for (x, y, xp, yp) in rows:
        out.append("%.7f %.7f %.7f %.7f" % (x, y, xp, yp))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
