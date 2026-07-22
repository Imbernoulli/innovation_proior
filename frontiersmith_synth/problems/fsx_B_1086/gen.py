#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE shared-addition-chain instance to stdout.

Instance schema:
  line 1: K                number of targets
  line 2: t_1 ... t_K      distinct target integers, 2 <= t_i <= 4000

The solver must output a straight-line program of additions from r_0 = 1 whose
registers contain every target; cost = number of instructions.

Planted structure (the trap): most targets lie on hidden "rails" t = R*u (+ c)
for a common rail R, multipliers u drawn from a small hidden addition chain, and
optionally a common offset c; a few noise targets hide the pattern. Per-target
binary chains (with or without prefix sharing) cannot see the rails, while a
joint (vectorial) addition-chain search reuses the rail registers across all
targets. Difficulty ladder 1..10 grows K, T, rail length and noise.
Deterministic: seeded only by testId.
"""
import sys
import random

LADDER = {
    1: dict(K=5, T=300, umax=9, offset=0, noise=1),
    2: dict(K=6, T=500, umax=11, offset=1, noise=1),
    3: dict(K=8, T=800, umax=13, offset=0, noise=1),
    4: dict(K=10, T=1200, umax=15, offset=1, noise=2),
    5: dict(K=11, T=1600, umax=16, offset=0, noise=2),
    6: dict(K=12, T=2000, umax=18, offset=1, noise=2),
    7: dict(K=13, T=2500, umax=20, offset=0, noise=2),
    8: dict(K=14, T=3000, umax=22, offset=1, noise=3),
    9: dict(K=15, T=3500, umax=24, offset=0, noise=3),
    10: dict(K=16, T=4000, umax=26, offset=1, noise=3),
}


def mini_chain(rng, umax, cnt):
    """A hidden small addition chain of multiplier values <= umax."""
    vals = {1}
    tries = 0
    while (len(vals) < cnt or max(vals) < umax) and tries < 10000:
        tries += 1
        pool = sorted(vals)
        a = rng.choice(pool)
        b = rng.choice(pool)
        v = a + b
        if 1 < v <= umax:
            vals.add(v)
    return sorted(vals)


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid not in LADDER:
        tid = ((tid - 1) % 10) + 1
    p = LADDER[tid]
    rng = random.Random(777000 + tid)
    K, T, umax, offset, noise = p["K"], p["T"], p["umax"], p["offset"], p["noise"]
    n_rail = K - noise

    U_all = mini_chain(rng, umax, n_rail + 2)
    pool = [u for u in U_all if u > 1]
    take = min(n_rail - 1, len(pool))
    U = set(rng.sample(pool, take)) if take > 0 else set()
    U.add(max(pool))
    U = sorted(U)
    while len(U) < n_rail:
        cand = rng.randint(2, umax)
        if cand not in U:
            U.append(cand)
            U.sort()

    R = rng.randint(max(3, T // (umax * 2)), max(4, T // umax))
    c = rng.randint(1, R - 1) if offset and R > 2 else 0

    targets = set()
    for u in U:
        t = R * u + c
        if 2 <= t <= T:
            targets.add(t)

    # noise targets that do NOT accidentally lie on the rails
    guard = 0
    while len(targets) < K and guard < 5000:
        guard += 1
        v = rng.randint(max(2, T // 3), T)
        if v in targets:
            continue
        if (v - c) % R == 0:
            continue
        targets.add(v)
    v = 2
    while len(targets) < K:  # fallback fill (never triggers on the ladder)
        if v not in targets:
            targets.add(v)
        v += 1

    out = sorted(targets)
    sys.stdout.write("%d\n%s\n" % (len(out), " ".join(str(t) for t in out)))


if __name__ == "__main__":
    main()
