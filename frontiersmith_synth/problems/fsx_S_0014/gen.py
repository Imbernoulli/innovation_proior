import sys, random

def main():
    tid = int(sys.argv[1])
    rng = random.Random(140014 + tid * 2711)

    # ---- difficulty / structure ladder (medium scale) ----
    if tid == 1:
        n = 12                      # tiny (example-scale sanity)
    else:
        n = 30 + 48 * (tid - 1)     # 78, 126, ... up to ~460 at tid 10 (=462 -> clamp)
    n = min(n, 460)

    # service radius grows with difficulty
    if tid <= 3:
        r = 1
    elif tid <= 7:
        r = 2
    else:
        r = 3

    # ---- build a connected graph: random spanning tree + cross-pipes (cycles) ----
    edges = set()
    def add(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        edges.add((a, b))

    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    for i in range(1, n):
        # attach perm[i] to a random earlier node -> spanning tree (connected)
        j = rng.randrange(i)
        add(perm[i], perm[j])

    # extra cross-pipes create cycles (general graph => covering is NP-hard, not a tree)
    extra = int(n * (0.20 + 0.06 * tid))
    cap = 3 * n - (n - 1)
    extra = min(extra, cap)
    attempts = 0
    while len([e for e in edges]) < (n - 1) + extra and attempts < 20 * (extra + 1):
        a = rng.randint(1, n)
        b = rng.randint(1, n)
        add(a, b)
        attempts += 1

    edge_list = list(edges)
    m = len(edge_list)

    # ---- degrees, for cost skew (make hubs expensive) ----
    deg = [0] * (n + 1)
    for a, b in edge_list:
        deg[a] += 1
        deg[b] += 1

    # top ~15% highest-degree tanks are pricey hubs -> naive max-coverage greedy overpays,
    # cost-aware strategies win -> genuine strategy divergence
    order = sorted(range(1, n + 1), key=lambda v: (-deg[v], v))
    hub_cut = max(1, int(0.15 * n))
    hub = set(order[:hub_cut])

    cost = [0] * (n + 1)
    for v in range(1, n + 1):
        base = rng.randint(1, 8)
        if v in hub:
            base += rng.randint(40, 90)
        # occasional random expensive non-hub to add noise
        elif rng.random() < 0.08:
            base += rng.randint(20, 50)
        cost[v] = min(100, base)

    # ---- emit ----
    out = []
    out.append("%d %d %d" % (n, m, r))
    out.append(" ".join(str(cost[v]) for v in range(1, n + 1)))
    for a, b in edge_list:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
