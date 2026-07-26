#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of "Quarry Foreman" (epoch-boundary-carver).

A cliff face [0, L) must be cut into slabs (segments) for lifting.  Every meter
x carries a posted rock-hardness reading h[x] (a tempting but decoy density
signal).  A fixed trace of Q lift-inspection passes (l_i, r_i, w_i) is given;
each pass re-touches every slab it overlaps at price w_i per slab.

Planted structure (the trap): a set of COARSE event points at IRREGULAR
spacing, confined to a randomly placed "event zone" covering roughly 35-55%
of the cliff, plus a nested cluster of FINE event points (tight spacing)
inside one coarse gap -- true multi-scale changepoints.  Most queries'
endpoints snap to these event points (coarse queries span one coarse gap,
fine queries span two nearby fine points, a residual of "noise" queries are
placed uniformly at random).  The hardness array h[] is a flat baseline plus
a few Gaussian mass bumps whose centers are confined to the COMPLEMENT of
the event zone -- so partitioning by hardness *mass* (the obvious
density-based reflex) starves exactly the region that carries the real
touch-cost signal.

Determinism: all randomness comes from random.Random(testId * 1000003 + 1149).
"""
import sys
import math
import random

GAMMA = 1.5  # fixed superlinear build exponent (also hardcoded in verify.py)

#                 1    2    3     4     5     6     7     8      9      10
L_LIST     = [300, 500, 900, 1500, 2500, 4000, 6000, 8500, 12000, 17000]
Q_LIST     = [12,  20,  30,  45,   60,   80,   105,  135,  170,   200]
# desired build-cost-only-optimal segment count, calibrated to be well ABOVE
# the round(sqrt(Q)) bucket count a "sqrt decomposition" reflex would pick
# (~3, 4, 5, 7, 8, 9, 10, 12, 13, 14) -- the true trade-off needs many more,
# smaller slabs than that generic heuristic ever considers.
KKARGET    = [9,   12,  15,  18,   21,   27,   33,   39,   48,    57]


def plant_events(rng, L):
    """Coarse event points at IRREGULAR spacing, confined to a randomly placed
    EVENT ZONE covering roughly 35-55% of the cliff, with a nested FINE
    cluster inside one coarse gap -- true multi-scale changepoints. The
    complement of the zone is where the decoy hardness mass will live."""
    zw = max(20, int(L * rng.uniform(0.35, 0.55)))
    zlo = rng.randint(0, max(0, L - zw))
    zhi = min(L, zlo + zw)

    ncoarse = rng.randint(5, 8)
    minsep = max(3, (zhi - zlo) // (4 * ncoarse))
    avg_gap = max(minsep + 1, int(1.7 * (zhi - zlo) / ncoarse))
    coarse = []
    cur = zlo + rng.randint(1, max(2, (zhi - zlo) // 20))
    for _ in range(ncoarse):
        if cur >= zhi - 2:
            break
        coarse.append(cur)
        gap = rng.randint(minsep, avg_gap)  # irregular: never evenly divided
        cur += gap
    coarse = sorted(set(c for c in coarse if 1 <= c <= L - 2))
    if len(coarse) < 2:
        coarse = [max(1, zlo + (zhi - zlo) // 3), min(L - 1, zlo + 2 * (zhi - zlo) // 3)]

    sidx = rng.randrange(len(coarse) - 1)
    lo_f, hi_f = coarse[sidx], coarse[sidx + 1]
    window = max(3, (hi_f - lo_f) // 6)
    center = rng.randint(lo_f + window, hi_f - window) if hi_f - window > lo_f + window else (lo_f + hi_f) // 2
    nfine = rng.randint(4, 7)
    span = list(range(max(1, center - window), min(L - 1, center + window) + 1))
    if len(span) < nfine:
        nfine = len(span)
    fine = sorted(rng.sample(span, nfine)) if nfine >= 1 else [center]
    return coarse, fine, (zlo, zhi)


def make_hardness(rng, L, zone):
    """Decoy density profile: a flat low baseline plus a few Gaussian mass
    bumps whose centers are confined to the COMPLEMENT of the event zone --
    so hardness *mass* concentrates exactly where the true changepoints are
    NOT, making equal-mass partitioning systematically starve the region
    that actually matters for touch cost. Positive ints, 1..20."""
    zlo, zhi = zone
    outside = [(a, b) for (a, b) in [(0, zlo), (zhi, L)] if b - a > 5]
    if not outside:
        outside = [(0, L)]
    h = [1.5] * L
    nbumps = rng.randint(2, 4)
    weights = [b - a for (a, b) in outside]
    for _ in range(nbumps):
        a, b = rng.choices(outside, weights=weights, k=1)[0]
        c = rng.uniform(a, b)
        sigma = max(3.0, (b - a) * rng.uniform(0.06, 0.16))
        amp = rng.uniform(12.0, 18.0)
        for x in range(L):
            d = x - c
            if abs(d) > 6 * sigma:
                continue
            h[x] += amp * math.exp(-(d * d) / (2 * sigma * sigma))
    out = []
    for x in range(L):
        v = h[x] + rng.uniform(-0.4, 0.4)
        iv = max(1, min(20, int(round(v))))
        out.append(iv)
    return out


def make_queries(rng, L, Q, coarse, fine):
    n_coarse_q = int(round(Q * 0.55))
    n_fine_q = int(round(Q * 0.30))
    n_noise_q = Q - n_coarse_q - n_fine_q

    queries = []

    def clamp(v):
        return max(0, min(L, v))

    for _ in range(n_coarse_q):
        # span a SINGLE coarse gap (occasionally two) -- not the whole cliff --
        # so precise alignment can genuinely zero out this query's touch cost
        if len(coarse) >= 2:
            a = rng.randrange(len(coarse) - 1)
            span = 2 if (rng.random() < 0.25 and a + 2 < len(coarse)) else 1
            b = a + span
        else:
            a, b = 0, 0
        pa, pb = coarse[a], coarse[b]
        l = clamp(pa + rng.randint(-2, 2))
        r = clamp(pb + rng.randint(-2, 2))
        if l > r:
            l, r = r, l
        if l == r:
            r = clamp(l + 1) if l + 1 <= L else l
            l = max(0, r - 1)
        w = rng.randint(60, 260)
        queries.append((l, r, w))

    for _ in range(n_fine_q):
        if len(fine) >= 2:
            a, b = rng.sample(range(len(fine)), 2)
        else:
            a, b = 0, 0
        pa, pb = fine[a], fine[b]
        l = clamp(pa + rng.randint(-1, 1))
        r = clamp(pb + rng.randint(-1, 1))
        if l > r:
            l, r = r, l
        if l == r:
            r = clamp(l + 1) if l + 1 <= L else l
            l = max(0, r - 1)
        w = rng.randint(60, 260)
        queries.append((l, r, w))

    for _ in range(n_noise_q):
        width = rng.randint(1, max(2, L // 10))
        l = rng.randint(0, max(0, L - 1))
        r = clamp(l + width)
        if r <= l:
            r = min(L, l + 1)
        w = rng.randint(1, 8)
        queries.append((l, r, w))

    rng.shuffle(queries)
    return queries[:Q]


def gen(test_id):
    rng = random.Random(test_id * 1000003 + 1149)
    i = (test_id - 1) % 10
    L = L_LIST[i]
    Q = Q_LIST[i]
    Kt = KKARGET[i]

    coarse, fine, zone = plant_events(rng, L)
    h = make_hardness(rng, L, zone)
    queries = make_queries(rng, L, Q, coarse, fine)

    # calibrate BASE so that the build-only-optimal segment count (ignoring
    # touch cost entirely, uniform split) is approximately Kt:
    #   K* = L * (0.5 / BASE)^(1/GAMMA)   =>   BASE = 0.5 / (Kt / L)^GAMMA
    ratio = Kt / L
    BASE = 0.5 / (ratio ** GAMMA)
    BASE = max(1, int(round(BASE)))

    return L, Q, BASE, h, queries


def main():
    test_id = int(sys.argv[1])
    L, Q, BASE, h, queries = gen(test_id)
    out = [f"{L} {Q} {BASE}"]
    out.append(" ".join(map(str, h)))
    for (l, r, w) in queries:
        out.append(f"{l} {r} {w}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
