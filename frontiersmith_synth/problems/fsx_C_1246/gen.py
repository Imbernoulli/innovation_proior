#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE ann-index-build instance to stdout.

Format (whitespace-separated tokens):
    N M R
    x_1 y_1        (N lines: dataset points, integer coords)
    ...
    x_N y_N
    Q
    qx_1 qy_1      (Q lines: held-out query points)
    ...
    qx_Q qy_Q

testId 1..10 is a hand-authored difficulty/trap ladder:
  1-3: a single diffuse point cloud (no cluster structure) -- warm-up cases
       where a plain nearest-neighbour graph already has everything it
       needs (no missing long-range reach is possible with only one
       region), so the exact-neighbour recipe cleanly beats an
       index-order/no-geometry construction.
  4-10: points drawn from K well-separated clusters (inter-cluster spacing
        >> intra-cluster jitter), with held-out queries spread across ALL
        clusters (not just the largest). 8 and 10 additionally make one
        cluster dominate the population (an "uneven" case) so that any
        entry-point rule that is not spatially aware locks onto the big
        cluster and starves the small ones. Seeded via testId only.
"""
import sys, random, math

CASES = {
    #      N,   K, spacing, jitter,  M, R,  Q, uneven
    1:  dict(N=60,  K=1, spacing=0,    jitter=250, M=7, R=2, Q=35, uneven=False),
    2:  dict(N=60,  K=1, spacing=0,    jitter=400, M=5, R=4, Q=25, uneven=False),
    3:  dict(N=60,  K=1, spacing=0,    jitter=200, M=6, R=4, Q=40, uneven=False),
    4:  dict(N=70,  K=3, spacing=3000, jitter=180, M=6, R=3, Q=21, uneven=False),
    5:  dict(N=85,  K=4, spacing=3200, jitter=170, M=6, R=4, Q=24, uneven=False),
    6:  dict(N=100, K=4, spacing=3500, jitter=150, M=7, R=4, Q=28, uneven=False),
    7:  dict(N=120, K=5, spacing=3600, jitter=150, M=7, R=4, Q=32, uneven=False),
    8:  dict(N=140, K=5, spacing=4000, jitter=150, M=7, R=5, Q=36, uneven=True),
    9:  dict(N=160, K=6, spacing=4200, jitter=140, M=8, R=5, Q=40, uneven=False),
    10: dict(N=180, K=7, spacing=4500, jitter=130, M=8, R=6, Q=45, uneven=True),
}


def cluster_sizes(N, K, uneven, rng):
    if K == 1:
        return [N]
    if not uneven:
        base = N // K
        sizes = [base] * K
        for i in range(N - base * K):
            sizes[i] += 1
        return sizes
    # one dominant cluster (~58% of N), remainder split evenly across the rest
    big = max(K + 2, round(N * 0.58))
    rest = N - big
    ncomp = K - 1
    base = rest // ncomp
    sizes = [base] * ncomp
    for i in range(rest - base * ncomp):
        sizes[i] += 1
    sizes = [big] + sizes
    return sizes


def cluster_centers(K, spacing):
    if K == 1:
        return [(0, 0)]
    ncols = math.ceil(math.sqrt(K))
    centers = []
    for i in range(K):
        col = i % ncols
        row = i // ncols
        centers.append((col * spacing, row * spacing))
    return centers


def main():
    tid_raw = int(sys.argv[1])
    tid = ((tid_raw - 1) % 10) + 1
    c = CASES[tid]
    rng = random.Random(20260000 + tid_raw)  # seed derives only from testId -> reproducible

    N, K = c['N'], c['K']
    spacing, jitter = c['spacing'], c['jitter']
    M, R, Q = c['M'], c['R'], c['Q']

    centers = cluster_centers(K, spacing)
    sizes = cluster_sizes(N, K, c['uneven'], rng)

    points = []          # (x, y)
    point_cluster = []   # which cluster each point belongs to (generation-time only)
    for ci, (cx, cy) in enumerate(centers):
        for _ in range(sizes[ci]):
            x = cx + rng.randint(-jitter, jitter)
            y = cy + rng.randint(-jitter, jitter)
            points.append((x, y))
            point_cluster.append(ci)

    # Held-out queries: spread as evenly as possible across ALL clusters
    # (not proportional to cluster population) so small clusters still get
    # exercised even when they hold few dataset points.
    qbase = Q // K
    qsizes = [qbase] * K
    for i in range(Q - qbase * K):
        qsizes[i] += 1
    queries = []
    for ci, (cx, cy) in enumerate(centers):
        for _ in range(qsizes[ci]):
            qx = cx + rng.randint(-jitter, jitter)
            qy = cy + rng.randint(-jitter, jitter)
            queries.append((qx, qy))

    out = []
    out.append(f"{N} {M} {R}")
    for (x, y) in points:
        out.append(f"{x} {y}")
    out.append(f"{len(queries)}")
    for (x, y) in queries:
        out.append(f"{x} {y}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
