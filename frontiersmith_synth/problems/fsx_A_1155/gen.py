#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of the Scrip-Ward market to stdout.
Deterministic: all randomness seeded purely from testId.
"""
import sys, random

ALPHA_NUM, ALPHA_DEN = 1, 2   # fixed bidding rule constants (same meaning every test)
TAX_DEN = 1000

# (N, R, k_active, v_active_range, v_mild_range, S_per_nurse_multiplier)
SPECS = {
    1:  (4,   5,  0, (100, 300), (100, 300), 400),
    2:  (6,   8,  0, (80,  320), (80,  320), 450),
    3:  (8,   10, 1, (500, 700), (60,  300), 450),
    4:  (10,  14, 1, (700, 950), (30,  90),  500),
    5:  (14,  20, 2, (750, 980), (25,  85),  450),
    6:  (20,  28, 3, (760, 990), (20,  80),  420),
    7:  (28,  36, 3, (770, 990), (20,  75),  400),
    8:  (40,  48, 4, (780, 1000),(15,  70),  380),
    9:  (60,  52, 5, (790, 1000),(15,  65),  360),
    10: (100, 60, 7, (800, 1000),(10,  60),  340),
}


def main():
    testId = int(sys.argv[1])
    N, R, k_active, va, vm, Smul = SPECS[testId]
    rnd = random.Random(1000 + testId * 97)
    S = N * Smul

    regime_change = (testId == 10)  # two disjoint active clusters, first/second half of rounds
    if k_active > 0:
        pool = list(range(N))
        rnd.shuffle(pool)
        if regime_change:
            clusterA = set(pool[:k_active])
            clusterB = set(pool[k_active:2 * k_active])
        else:
            active = set(pool[:k_active])
    else:
        active = set()

    out = []
    out.append(f"{N} {R}")
    out.append(f"{ALPHA_NUM} {ALPHA_DEN} {S} {TAX_DEN}")
    for r in range(R):
        if k_active > 0:
            if regime_change:
                cur_active = clusterA if r < R // 2 else clusterB
            else:
                cur_active = active
        else:
            cur_active = set()
        row = []
        for i in range(N):
            if i in cur_active:
                row.append(rnd.randint(*va))
            else:
                row.append(rnd.randint(*vm))
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
