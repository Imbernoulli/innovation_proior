import sys, random

def main():
    tid = int(sys.argv[1])
    rng = random.Random(770077 + tid * 101)

    # ---- difficulty / structure ladder ----
    sizes = [16, 24, 40, 60, 90, 120, 160, 200, 260, 300]
    n = sizes[min(tid, 10) - 1]
    if n % 2 != 0:
        n += 1

    # planted balanced bipartition y (RANDOM over indices -> uncorrelated with the
    # index-block reference the checker uses as its baseline).
    y = [0] * (n // 2) + [1] * (n - n // 2)
    rng.shuffle(y)                       # y[v-1] is the "true" station of junction v
    grp0 = [v for v in range(1, n + 1) if y[v - 1] == 0]
    grp1 = [v for v in range(1, n + 1) if y[v - 1] == 1]

    # target edge counts: heavy inter-group cables dominate; a lighter layer of
    # intra-group noise cables creates local optima (so greedy != strong).
    inter_pairs = len(grp0) * len(grp1)
    intra_pairs = (len(grp0) * (len(grp0) - 1) + len(grp1) * (len(grp1) - 1)) // 2
    HE = min(5 * n, int(0.55 * inter_pairs))
    LE = min(2 * n, int(0.55 * intra_pairs))

    edges = {}   # (a,b) with a<b -> w

    def add_edge(u, v, w):
        if u == v:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edges:
            return False
        edges[(a, b)] = w
        return True

    # heavy inter-group cables (weight 6..10): the "signal" a good cut collects.
    added, attempts = 0, 0
    while added < HE and attempts < HE * 40 + 200:
        attempts += 1
        u = rng.choice(grp0); v = rng.choice(grp1)
        if add_edge(u, v, rng.randint(6, 10)):
            added += 1

    # light intra-group cables (weight 1..3): "noise" trunks that the optimum leaves
    # internal; they trap single-pass heuristics in inferior local optima.
    added, attempts = 0, 0
    while added < LE and attempts < LE * 40 + 200:
        attempts += 1
        g = grp0 if rng.random() < 0.5 else grp1
        if len(g) < 2:
            continue
        u, v = rng.sample(g, 2)
        if add_edge(u, v, rng.randint(1, 3)):
            added += 1

    # guarantee at least one edge (so B >= 1)
    if not edges:
        add_edge(1, n, rng.randint(6, 10))

    items = list(edges.items())
    rng.shuffle(items)                   # scramble line order
    m = len(items)

    out = ["%d %d" % (n, m)]
    for (a, b), w in items:
        # randomly flip endpoint order so u<v is not always implied
        if rng.random() < 0.5:
            out.append("%d %d %d" % (a, b, w))
        else:
            out.append("%d %d %d" % (b, a, w))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
