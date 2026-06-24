import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(2, 7)
    # Build a connected graph: start with a random spanning tree, then add extra edges.
    perm = list(range(n))
    rng.shuffle(perm)
    edges = []
    cap_hi = rng.choice([5, 9, 20, 100])
    for i in range(1, n):
        u = perm[i]
        v = perm[rng.randint(0, i - 1)]
        c = rng.randint(1, cap_hi)
        edges.append((u, v, c))

    # extra random edges (possibly parallel between same pair, different caps)
    extra = rng.randint(0, n * 2)
    for _ in range(extra):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        c = rng.randint(1, cap_hi)
        edges.append((u, v, c))

    rng.shuffle(edges)
    m = len(edges)

    # queries (s != t)
    q = rng.randint(1, 6)
    queries = []
    for _ in range(q):
        s = rng.randint(0, n - 1)
        t = rng.randint(0, n - 1)
        while t == s:
            t = rng.randint(0, n - 1)
        queries.append((s, t))

    lines = [f"{n} {m} {q}"]
    for (u, v, c) in edges:
        lines.append(f"{u} {v} {c}")
    for (s, t) in queries:
        lines.append(f"{s} {t}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
