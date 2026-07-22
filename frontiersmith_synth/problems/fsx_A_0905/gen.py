#!/usr/bin/env python3
"""
gen.py -- deterministic generator for fsx_A_0905 (mosaic-gradient-tessera).

Emits one instance:
  line 1: n c K
  line 2: v_1 .. v_c          (strictly increasing tile values)
  line 3: cnt_1 .. cnt_c      (exact tile multiset, sum = n*n)
  line 4: Tcenter Tedge       (radial gradient endpoints)

The multiset cnt[] is FIXED to exactly match the diagonal residue class
sizes (count of cells with (i+j) mod c == k for the n x n grid) -- this is
what lets the "obvious" cap-safe blind construction (trivial) be a pure,
zero-adaptivity checkerboard-style formula instead of a search, while still
leaving a genuine transportation problem for anyone who wants to chase the
gradient. testId is a difficulty/structure ladder: 1 tiny sanity case,
growing to large adversarial trap cases by testId 10.  Only Python's
`random` seeded by testId is used -> fully deterministic.
"""
import random
import sys


def emit(n, c, K, v, cnt, tcenter, tedge):
    assert len(v) == c and len(cnt) == c
    assert sum(cnt) == n * n, (sum(cnt), n * n)
    assert all(cnt_i >= 0 for cnt_i in cnt)
    assert all(v[i] < v[i + 1] for i in range(c - 1))
    out = []
    out.append(f"{n} {c} {K}")
    out.append(" ".join(str(x) for x in v))
    out.append(" ".join(str(x) for x in cnt))
    out.append(f"{tcenter} {tedge}")
    sys.stdout.write("\n".join(out) + "\n")


def diagonal_residue_counts(n, c):
    cnt = [0] * c
    for i in range(n):
        for j in range(n):
            cnt[(i + j) % c] += 1
    return cnt


def gen_values(rng, c, vmax=900, kind="spread"):
    if kind == "clustered":
        # most colors clustered near one end, one or two outliers far away --
        # a harder palette to hit precisely (bigger quantization jumps)
        lo = rng.sample(range(0, vmax // 3), max(1, c - 2))
        hi = rng.sample(range(2 * vmax // 3, vmax + 1), c - len(lo))
        pool = sorted(set(lo) | set(hi))
        while len(pool) < c:
            cand = rng.randrange(0, vmax + 1)
            if cand not in pool:
                pool.append(cand)
                pool.sort()
        return pool[:c]
    pool = rng.sample(range(0, vmax + 1), c)
    pool.sort()
    return pool


def main():
    testId = int(sys.argv[1])
    rng = random.Random(1_000_003 * testId + 7)

    # difficulty ladder: (n, c, K, value-kind, Tcenter/Tedge mode)
    #   K is deliberately tight (checkerboard-safe minimum + a hair of slack)
    #   so the run-cap genuinely binds against gradient-chasing.
    plan = {
        1: (5, 4, 2, "spread", "wide"),
        2: (8, 4, 1, "spread", "wide"),
        3: (12, 5, 2, "spread", "wide"),
        4: (16, 5, 1, "clustered", "wide"),   # trap: hard palette, tight cap
        5: (20, 6, 2, "spread", "narrow"),    # planted: gentle gradient
        6: (24, 6, 1, "spread", "wide"),       # trap: tight cap, large n
        7: (30, 7, 2, "clustered", "wide"),   # trap: hard palette
        8: (36, 7, 1, "spread", "wide"),      # trap: tightest cap, large n
        9: (45, 8, 2, "spread", "wide"),
        10: (54, 8, 1, "clustered", "wide"),  # trap: largest + hardest palette
    }
    n, c, K, vkind, tmode = plan[testId]

    v = gen_values(rng, c, kind=vkind)
    cnt = diagonal_residue_counts(n, c)

    if tmode == "wide":
        tcenter = v[0] if rng.random() < 0.5 else v[-1]
        tedge = v[-1] if tcenter == v[0] else v[0]
    else:
        lo_idx = rng.randrange(0, c // 2 + 1)
        hi_idx = rng.randrange(c // 2, c)
        if hi_idx <= lo_idx:
            hi_idx = min(c - 1, lo_idx + 1)
        tcenter, tedge = v[lo_idx], v[hi_idx]

    emit(n, c, K, v, cnt, tcenter, tedge)


if __name__ == "__main__":
    main()
