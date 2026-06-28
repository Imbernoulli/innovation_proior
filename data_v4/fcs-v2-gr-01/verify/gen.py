import sys, random

# Random small min-cost-flow instance with possibly-negative edge costs and
# guaranteed NO negative-cost cycle (the problem's standing assumption).
#
# We guarantee acyclicity-of-negativity by assigning each vertex a random
# integer "level" p[v] and only emitting an edge u->v whose cost w satisfies
# w + p[u] - p[v] >= 0  (i.e. w >= p[v] - p[u]). Any cycle then has total cost
# = sum of (w + p[u]-p[v]) >= 0, so no negative cycle can exist, while edges
# can still individually be negative.

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(2, 7)
    # random potentials define the "feasible" cost floor per edge
    p = [rng.randint(-5, 5) for _ in range(n)]

    s = rng.randint(0, n - 1)
    t = rng.randint(0, n - 1)
    while t == s:
        t = rng.randint(0, n - 1)

    max_m = min(14, n * (n - 1))
    m = rng.randint(0, max_m)

    edges = []
    seen = set()
    attempts = 0
    while len(edges) < m and attempts < 200:
        attempts += 1
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        cap = rng.randint(0, 6)
        floor = p[v] - p[u]               # reduced cost must be >= 0
        w = floor + rng.randint(0, 8)     # so cost can be negative but no neg-cycle
        edges.append((u, v, cap, w))

    m = len(edges)
    # F up to a bit beyond plausible max-flow so IMPOSSIBLE cases appear too.
    F = rng.randint(0, 8)

    out = [f"{n} {m} {s} {t} {F}"]
    for (u, v, cap, w) in edges:
        out.append(f"{u} {v} {cap} {w}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
