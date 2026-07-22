#!/usr/bin/env python3
"""gen.py <testId> -- prints one instance of the Vesting-Lock-in Allocator problem.

Deterministic: seeded purely from testId. Plants "decoy vs riser" trap pairs:
a decoy instrument has an enticing single-period spike right now but a rotten
window average once its long lock is honored; a riser instrument looks mediocre
right now but opens a sustained high-rate window a couple of periods later that
lines up with a shorter lock. Chasing the current best rate walks straight into
the decoy and misses the riser's window entirely.
"""
import sys, random

SIZES = {
    1: (6, 2), 2: (8, 3), 3: (10, 3), 4: (14, 4), 5: (18, 5),
    6: (24, 5), 7: (30, 6), 8: (36, 6), 9: (44, 7), 10: (50, 8),
}


def build(test_id: int):
    T, K = SIZES[test_id]
    rnd = random.Random(1000003 * test_id + 7)
    C0 = 1_000_000

    # baseline lock durations (moderate) and steady positive background rates (bp)
    L = [rnd.randint(2, max(2, T // 4)) for _ in range(K)]
    rate = [[rnd.randint(100, 400) for _ in range(K)] for _ in range(T)]  # rate[t-1][k-1]

    n_pairs = 1 if T <= 14 else 2
    order = list(range(K))
    rnd.shuffle(order)

    block = max(9, T // max(1, n_pairs))
    for i in range(n_pairs):
        d_idx = order[(2 * i) % K]
        r_idx = order[(2 * i + 1) % K]
        lo = 1 + i * block
        hi = min(T - 7, lo + max(0, block - 9))
        p0 = rnd.randint(lo, hi) if hi >= lo else max(1, T - 7)
        p0 = max(1, min(p0, T - 7))

        # riser's short lock, decided first so the decoy lock can be made to outlast it
        Lr = rnd.randint(3, 5)

        # decoy: a lock long enough to swallow the riser's whole window, huge spike
        # right now, then a mild multi-period slump (below the noise floor, but not
        # catastrophic -- the cost is the missed riser window, not a crash).
        Ld = rnd.randint(max(Lr + 3, 6), max(Lr + 3, 6) + 3)
        Ld = min(Ld, T - p0 + 1)
        L[d_idx] = Ld
        rate[p0 - 1][d_idx] = rnd.randint(1500, 2000)
        for tt in range(p0, min(p0 + Ld - 1, T)):
            rate[tt][d_idx] = rnd.randint(-300, 50)

        # riser: unremarkable now (ordinary noise level), strong sustained window
        # starting p0+2 that a decoy-locked competitor cannot reach.
        L[r_idx] = Lr
        wstart = p0 + 2
        for tt in range(wstart - 1, min(wstart - 1 + Lr, T)):
            rate[tt][r_idx] = rnd.randint(600, 900)

    lines = [f"{T} {K}", f"{C0}", " ".join(str(x) for x in L)]
    for t in range(T):
        lines.append(" ".join(str(x) for x in rate[t]))
    return "\n".join(lines) + "\n"


def main():
    test_id = int(sys.argv[1])
    sys.stdout.write(build(test_id))


if __name__ == "__main__":
    main()
