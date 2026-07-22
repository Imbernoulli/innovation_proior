#!/usr/bin/env python3
"""gen.py <testId> -> prints one instance of the levee-budget/hydrograph-sweep
problem to stdout. Deterministic: all randomness is seeded from testId only.
"""
import sys
import random

EMAX = 60
K = 24
SIZES = [24, 36, 48, 70, 95, 125, 155, 185, 220, 260]
TRAP_IDS = {3, 6, 9}


def build(test_id):
    rng = random.Random(20260731 + test_id * 977)
    N = SIZES[test_id - 1] if 1 <= test_id <= len(SIZES) else SIZES[-1]
    is_trap = test_id in TRAP_IDS

    e = [0] * N
    wall_w = min(3, max(1, N // 12))
    for i in range(wall_w):
        e[i] = rng.randint(EMAX - 6, EMAX - 1)
        e[N - 1 - i] = rng.randint(EMAX - 6, EMAX - 1)

    lo, hi = wall_w, N - 1 - wall_w
    cur = rng.randint(3, 10)
    for i in range(lo, hi + 1):
        step = rng.randint(-4, 4)
        cur = max(0, min(EMAX - 10, cur + step))
        e[i] = cur

    # a handful of internal ridges to create a nested basin structure
    n_ridges = max(2, N // 45)
    pool = list(range(lo + 3, hi - 2)) if hi - 2 > lo + 3 else []
    rng.shuffle(pool)
    ridge_positions = sorted(pool[:min(n_ridges, len(pool))])
    for rp in ridge_positions:
        hr = rng.randint(14, EMAX - 12)
        e[rp] = hr
        for d in (-1, 1):
            if lo <= rp + d <= hi:
                e[rp + d] = min(e[rp + d], hr - 1)

    v = [rng.randint(1, 15) for _ in range(N)]
    n_assets = max(1, N // 90)
    if n_assets > 0:
        asset_positions = rng.sample(range(lo, hi + 1), min(n_assets, hi - lo + 1))
        for ap in asset_positions:
            v[ap] = rng.randint(120, 300)

    saddle = None
    collector_start = collector_end = None
    cluster_start = cluster_end = None
    if is_trap:
        # an enclosed collector basin (walled in by the domain's own tall
        # edges), a cheap-to-raise low-value saddle, and a high-value
        # cluster just beyond it -- the collector only overflows past the
        # saddle under the sweep's bigger storms, at which point the
        # overflow reroutes straight into the valuable cluster.
        collector_len = max(6, N // 8)
        collector_start = lo
        collector_end = min(hi - 8, collector_start + collector_len - 1)
        for i in range(collector_start, collector_end + 1):
            e[i] = rng.randint(1, 5)
            v[i] = 1

        saddle = min(hi - 6, collector_end + 1)
        e[saddle] = rng.randint(16, 20)
        v[saddle] = 1
        for d in (-2, -1, 1, 2):
            if lo <= saddle + d <= hi:
                e[saddle + d] = min(e[saddle + d], e[saddle] - 1)

        cluster_len = max(6, N // 8)
        cluster_start = saddle + 2
        cluster_end = min(hi - 1, cluster_start + cluster_len - 1)
        for i in range(cluster_start, cluster_end + 1):
            e[i] = rng.randint(2, 7)
            v[i] = rng.randint(320, 620)

    Budget = int(round(N * 0.5)) + 12

    storms = []
    if is_trap:
        # rain falls ONLY on the collector -- the cluster must flood purely
        # via overflow through the saddle, never from its own direct rain,
        # or protecting the saddle would not matter
        a = collector_start
        b = collector_end
        cl = collector_end - collector_start + 1
        # sized so a modest saddle raise (well within budget) neutralizes
        # most of the overflow, but h=0 genuinely floods the cluster
        Vbig = int(cl * rng.uniform(28.0, 36.0))
        storms.append((a, b, Vbig))
        storms.append((a, b, int(Vbig * 0.75)))
        storms.append((a, b, int(Vbig * 1.35)))

    while len(storms) < K:
        wtype = rng.random()
        if wtype < 0.25:
            a, b = 0, N - 1
        elif wtype < 0.6:
            wlen = max(6, min(N, rng.randint(N // 4, max(N // 4 + 1, N // 2))))
            a = rng.randint(0, N - wlen)
            b = a + wlen - 1
        else:
            lo_w = max(4, N // 15)
            hi_w = max(lo_w + 1, N // 6)
            wlen = max(4, min(N, rng.randint(lo_w, hi_w)))
            a = rng.randint(0, N - wlen)
            b = a + wlen - 1
        span = b - a + 1
        # volume roughly proportional to window width, with variety -- kept
        # modest so the sweep's ordinary members don't drown out the
        # engineered chokepoint storms on trap instances
        V = int(span * rng.uniform(0.4, 2.0))
        storms.append((a, b, max(1, V)))
    storms = storms[:K]

    return N, Budget, K, e, v, storms


def main():
    test_id = int(sys.argv[1])
    N, Budget, K, e, v, storms = build(test_id)
    lines = []
    lines.append(f"{N} {Budget} {K}")
    lines.append(" ".join(map(str, e)))
    lines.append(" ".join(map(str, v)))
    for (a, b, V) in storms:
        lines.append(f"{a} {b} {V}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
