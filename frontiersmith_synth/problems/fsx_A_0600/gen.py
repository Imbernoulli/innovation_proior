import sys, random, math

# fsx_A_0600 -- epidemic-threshold-eigendrop  (Format C, minimize spectral radius)
# `python3 gen.py <testId>` prints ONE instance. testId is the ONLY randomness source.
#
# Planted structure so WEIGHT/DEGREE greedy is a moderate trap:
#   - CORE clique (nc nodes, moderate uniform weight wc): the spectral core; its edges
#     have the largest eigen-leverage  w * x_i * x_j  despite only moderate weight.
#   - PERIPHERY (npr nodes, each attached to 2 core nodes with weight wa): moderate but
#     clearly sub-core eigenvector centrality.
#   - HEAVY DECOYS (k+1 periphery<->periphery edges, weight wd > wc): the HIGHEST weights
#     in the graph, so weight-greedy cuts them first -- but their endpoints sit off the
#     core, so cutting them removes only ~40-50% of the eigen-drop the core edges give.
#   - REFERENCE edges (placed FIRST, indices 1..k): chosen adaptively so that removing
#     them yields ~ (eigen-greedy drop)/7.5, i.e. a weak-but-nonzero baseline. The checker
#     normalizes by exactly this first-k drop, so `trivial` (cut first k) scores ~0.1 and
#     the strong (eigen-leverage) solution lands well under the score cap.
#
# Difficulty grows with testId (more nodes, larger budget).

ITERS = 500
WC = 10          # core edge weight
WA = 6           # periphery attach weight (moderate centrality)
WD = 20          # heavy decoy weight (highest in graph -> weight-greedy bait)


def perron(n, elist):
    if n == 0:
        return []
    x = [1.0] * n
    s = math.sqrt(float(n))
    x = [v / s for v in x]
    for _ in range(ITERS):
        y = [0.0] * n
        for (i, j, w) in elist:
            y[i] += w * x[j]
            y[j] += w * x[i]
        nrm = math.sqrt(sum(v * v for v in y))
        if nrm <= 0.0:
            return x
        inv = 1.0 / nrm
        x = [v * inv for v in y]
    return x


def lam(n, elist):
    x = perron(n, elist)
    if not x:
        return 0.0
    y = [0.0] * n
    for (i, j, w) in elist:
        y[i] += w * x[j]
        y[j] += w * x[i]
    return sum(x[t] * y[t] for t in range(n))


def drop_of(n, edges, rm):
    l0 = lam(n, edges)
    rmset = set(rm)
    kept = [edges[i] for i in range(len(edges)) if i not in rmset]
    return l0 - lam(n, kept)


def eigen_greedy(n, edges, k):
    rem = set(range(len(edges)))
    removed = []
    for _ in range(k):
        cur = [edges[i] for i in rem]
        x = perron(n, cur)
        bi = -1
        bL = -1.0
        for idx in rem:
            i, j, w = edges[idx]
            L = w * x[i] * x[j]
            if L > bL:
                bL = L
                bi = idx
        if bi < 0:
            break
        removed.append(bi)
        rem.discard(bi)
    return removed


def build_raw(t):
    rng = random.Random(20260600 + t * 1009)
    k = 3 + (t % 4)          # 3..6
    nc = 6 + (t % 3)         # 6..8 core
    npr = 5 + (t % 4)        # 5..8 periphery
    nleaf = 1 + (t // 3)     # a few leaves for size

    used = set()
    edges = []

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def add(a, b, w):
        e = key(a, b)
        if e in used:
            return False
        used.add(e)
        edges.append((e[0], e[1], w))
        return True

    for i in range(nc):
        for j in range(i + 1, nc):
            add(i, j, WC + rng.randint(-1, 1))

    peri = list(range(nc, nc + npr))
    for p in peri:
        c1 = rng.randrange(nc)
        c2 = rng.randrange(nc)
        add(p, c1, WA + rng.randint(-1, 1))
        if c2 != c1:
            add(p, c2, WA + rng.randint(-1, 1))

    # heavy decoys (k+1) as periphery<->periphery edges
    nd = 0
    att = 0
    while nd < k + 1 and att < 500:
        att += 1
        a = peri[rng.randrange(npr)]
        b = peri[rng.randrange(npr)]
        if a == b:
            continue
        if add(a, b, WD + rng.randint(-2, 2)):
            nd += 1

    n = nc + npr
    for _ in range(nleaf):
        node = n
        n += 1
        add(peri[rng.randrange(npr)], node, 1 + rng.randint(0, 1))

    return n, k, edges


def choose_reference(n, edges, k, target):
    # Greedily assemble exactly k edges whose combined removal-drop is closest to `target`.
    chosen = []
    for _ in range(k):
        best = -1
        bestgap = None
        for idx in range(len(edges)):
            if idx in chosen:
                continue
            d = drop_of(n, edges, chosen + [idx])
            gap = abs(d - target)
            if bestgap is None or gap < bestgap:
                bestgap = gap
                best = idx
        chosen.append(best)
    return chosen


def build(t):
    n, k, edges = build_raw(t)
    strong = eigen_greedy(n, edges, k)
    d_strong = drop_of(n, edges, strong)
    target = d_strong / 7.5          # baseline drop -> strong ratio ~0.75-0.80
    ref = choose_reference(n, edges, k, target)

    # reorder: reference edges first (indices 1..k), then the rest
    refset = set(ref)
    order = list(ref) + [i for i in range(len(edges)) if i not in refset]
    edges = [edges[i] for i in order]
    return n, k, edges


def main():
    t = int(sys.argv[1])
    n, k, edges = build(t)
    m = len(edges)
    out = ["%d %d %d" % (n, m, k)]
    for (u, v, w) in edges:
        out.append("%d %d %d" % (u + 1, v + 1, w))   # 1-indexed nodes
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
