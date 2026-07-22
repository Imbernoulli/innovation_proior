#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints one instance of the "River Staircase" dispatch
problem to stdout.  testId in 1..10 is a difficulty ladder (small -> large,
steady -> flood-stressed).  All randomness is seeded ONLY from testId so the
instance is bit-for-bit reproducible.
"""
import sys, random

# (N reservoirs, T horizon) ladder, small -> large.
SIZES = [
    (4, 40), (4, 70), (5, 100), (5, 140), (6, 180),
    (6, 220), (7, 260), (7, 300), (8, 350), (8, 400),
]

# Test cases that get an engineered flood pulse into the headwater reservoir
# (trap cases: a myopic "always release at max flow" policy cannot both keep
# good head AND avoid forced spill through these).
TRAP_IDS = {3, 5, 7, 9, 10}


def gen(test_id: int) -> str:
    if not (1 <= test_id <= 10):
        raise ValueError("testId must be in 1..10")
    N, T = SIZES[test_id - 1]
    rng = random.Random(900000 + 97 * test_id)

    C = []
    Rmax = []
    A = []
    B = []
    DELAY = []
    INIT = []
    for i in range(N):
        c = rng.uniform(600.0, 1800.0)
        rmax = rng.uniform(0.15, 0.30) * c
        a = rng.uniform(0.10, 0.30)
        b = rng.uniform(0.55, 1.00)
        delay = 0 if i == 0 else rng.randint(1, 5)
        init = rng.uniform(0.35, 0.55) * c
        C.append(c); Rmax.append(rmax); A.append(a); B.append(b)
        DELAY.append(delay); INIT.append(init)

    inflow = [[0.0] * T for _ in range(N)]
    for i in range(N):
        if i == 0:
            base = rng.uniform(0.010, 0.030) * C[i]
        else:
            base = rng.uniform(0.003, 0.012) * C[i]
        for t in range(T):
            noise = rng.uniform(0.8, 1.2)
            inflow[i][t] = max(0.0, base * noise)

        if i == 0 and test_id in TRAP_IDS:
            # inject one flood pulse: a sustained multi-step surge that a
            # reservoir with no spare buffer cannot absorb without spilling.
            w = max(5, T // 10)
            start = rng.randint(T // 4, max(T // 4, T // 2))
            start = min(start, T - w - 1) if T - w - 1 > 0 else 0
            mult = rng.uniform(8.0, 14.0)
            for t in range(start, min(T, start + w)):
                inflow[i][t] *= mult
            if test_id in (9, 10):
                # second, later pulse for the biggest instances
                w2 = max(4, T // 14)
                start2 = min(T - w2 - 1, start + w + rng.randint(10, 25))
                if start2 > start + w:
                    mult2 = rng.uniform(6.0, 10.0)
                    for t in range(start2, min(T, start2 + w2)):
                        inflow[i][t] *= mult2

    out = [f"{N} {T}"]
    for i in range(N):
        out.append(f"{C[i]:.6f} {Rmax[i]:.6f} {A[i]:.6f} {B[i]:.6f} {DELAY[i]} {INIT[i]:.6f}")
    for i in range(N):
        out.append(" ".join(f"{v:.6f}" for v in inflow[i]))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    tid = int(sys.argv[1])
    sys.stdout.write(gen(tid))
