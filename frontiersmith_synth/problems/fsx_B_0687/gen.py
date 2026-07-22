#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE instance of the surge-roster-boundary-stagger problem.
Deterministic: everything derives from testId via a seeded RNG.
"""
import sys, math, random


def coverage(starts, L):
    cov = [0] * 24
    for s in starts:
        for k in range(L):
            cov[(s + k) % 24] += 1
    return cov


def allowed_hours(g, r):
    return [h for h in range(24) if h % g == r]


def uniform_pattern(A, W, offset=0):
    n = len(A)
    return [A[(offset + (i * n) // W) % n] for i in range(W)]


def main():
    testId = int(sys.argv[1])
    rng = random.Random(20260 + 97 * testId)

    # difficulty ladder: small -> large / adversarial
    # NOTE: W must NOT be an exact multiple of len(allowed_hours) = 24/g, or a
    # uniform-spread pattern over ALL rotations degenerates to the identical
    # coverage set for every offset (the phase trick becomes a no-op).
    sizes = [
        dict(W=6, L=5, C=4, K=6, g=1),
        dict(W=8, L=6, C=4, K=8, g=1),
        dict(W=10, L=7, C=4, K=10, g=2),
        dict(W=13, L=7, C=5, K=12, g=2),
        dict(W=14, L=8, C=5, K=14, g=1),
        dict(W=17, L=9, C=5, K=16, g=3),
        dict(W=18, L=9, C=5, K=18, g=1),
        dict(W=20, L=10, C=6, K=20, g=2),
        dict(W=23, L=10, C=6, K=23, g=1),
        dict(W=28, L=11, C=6, K=26, g=2),
    ]
    p = sizes[testId - 1]
    W, L, C, K, g = p["W"], p["L"], p["C"], p["K"], p["g"]
    r = rng.randrange(g)
    A = allowed_hours(g, r)

    # base arrival curve: gentle daily rhythm, moderate utilization so surges (not
    # base load alone) are what create backlog.
    avg_workers = (W * L) / 24.0
    avg_cap = C * avg_workers
    base = []
    for h in range(24):
        shape = 1.0 + 0.30 * math.sin(2 * math.pi * (h - 9) / 24.0)
        val = 0.5 * avg_cap * shape
        val += rng.uniform(-0.08, 0.08) * avg_cap
        base.append(max(0, round(val)))

    # compute the default (offset=0) uniform-spread roster's weak-coverage hours;
    # this is the "obvious" schedule a naive scheduler anchors at the first
    # allowed hour.
    default_starts = uniform_pattern(A, W, offset=0)
    cov0 = coverage(default_starts, L)
    reachable = set()
    for s in A:
        for k in range(L):
            reachable.add((s + k) % 24)
    m = min(cov0[h] for h in reachable)
    weak_hours = [h for h in reachable if cov0[h] == m]
    weak_hours.sort()

    cap_weak = C * m

    profiles = []
    n_trap = max(3, int(round(K * 0.7)))
    for _ in range(n_trap):
        sk = rng.choice(weak_hours)
        maxd = max(1, min(4, len(weak_hours)))
        dk = rng.randint(1, maxd)
        amp = rng.uniform(1.3, 2.1)
        ak = max(1, round(amp * max(cap_weak, 1)))
        profiles.append((sk, dk, ak))
    while len(profiles) < K:
        sk = rng.randrange(24)
        dk = rng.randint(1, 3)
        amp = rng.uniform(0.8, 1.6)
        ak = max(1, round(amp * max(cap_weak, 1)))
        profiles.append((sk, dk, ak))
    rng.shuffle(profiles)

    out = []
    out.append(f"{W} {L} {C} {g} {r}")
    out.append(" ".join(str(x) for x in base))
    out.append(str(K))
    for sk, dk, ak in profiles:
        out.append(f"{sk} {dk} {ak}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
