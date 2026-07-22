#!/usr/bin/env python3
"""gen.py <testId> -- prints one instance of "Shared Stockpot Chain" to stdout.

Instance = a chain of m base potions (a matrix-chain of dims d_0..d_m) plus Q
"brew orders" (queries), each a contiguous range [L,R) of the chain that must be
reduced to a single potion via pairwise combinations. Query ranges are generated
around a hidden set of shared "stockpot" anchor points with small per-query jitter,
so many queries overlap on common sub-ranges without being byte-identical --
this plants the cross-query reuse structure. All randomness is seeded from testId.
"""
import random
import sys

# (m, hub_stride, Q, max_span_hubs, jitter_max)
CASES = {
    1: dict(m=10, hub_stride=5, Q=4, max_span_hubs=2, jitter_max=1),
    2: dict(m=14, hub_stride=4, Q=7, max_span_hubs=2, jitter_max=1),
    3: dict(m=20, hub_stride=5, Q=10, max_span_hubs=3, jitter_max=2),
    4: dict(m=28, hub_stride=6, Q=14, max_span_hubs=3, jitter_max=2),
    5: dict(m=36, hub_stride=6, Q=20, max_span_hubs=3, jitter_max=2),
    6: dict(m=44, hub_stride=7, Q=26, max_span_hubs=3, jitter_max=3),
    7: dict(m=52, hub_stride=8, Q=32, max_span_hubs=3, jitter_max=4),
    8: dict(m=60, hub_stride=8, Q=38, max_span_hubs=3, jitter_max=4),
    9: dict(m=72, hub_stride=9, Q=45, max_span_hubs=3, jitter_max=4),
    10: dict(m=80, hub_stride=9, Q=50, max_span_hubs=3, jitter_max=4),
}


def build(test_id):
    cfg = CASES[test_id]
    rnd = random.Random(20260721 * 1000003 + test_id * 977 + 13)
    m = cfg["m"]
    hub_stride = cfg["hub_stride"]
    q_target = cfg["Q"]
    max_span_hubs = cfg["max_span_hubs"]
    jitter_max = cfg["jitter_max"]

    dims = [rnd.randint(20, 200) for _ in range(m + 1)]

    anchors = list(range(0, m + 1, hub_stride))
    if anchors[-1] != m:
        anchors.append(m)
    K = len(anchors)

    queries = []
    tries = 0
    seen = set()
    while len(queries) < q_target and tries < q_target * 80:
        tries += 1
        i = rnd.randrange(0, K - 1)
        span = rnd.randint(1, min(max_span_hubs, K - 1 - i))
        j = i + span
        core_l, core_r = anchors[i], anchors[j]
        jl = 0 if rnd.random() < 0.22 else rnd.randint(0, jitter_max)
        jr = 0 if rnd.random() < 0.22 else rnd.randint(0, jitter_max)
        L = max(0, core_l - jl)
        R = min(m, core_r + jr)
        if R - L < 2:
            continue
        queries.append((L, R))
        seen.add((L, R))

    # pad defensively if the random walk under-filled (keeps determinism: seeded rnd continues)
    while len(queries) < q_target:
        L = rnd.randrange(0, max(1, m - 2))
        R = min(m, L + 2 + rnd.randrange(0, hub_stride))
        if R - L >= 2:
            queries.append((L, R))
    queries = queries[:q_target]

    return m, dims, queries


def main():
    test_id = int(sys.argv[1])
    m, dims, queries = build(test_id)
    out = [str(m), " ".join(map(str, dims)), str(len(queries))]
    for L, R in queries:
        out.append(f"{L} {R}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
