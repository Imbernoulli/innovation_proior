import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 7)
    # allow a few edges; keep small so simple-path enumeration is cheap
    max_edges = n * n
    m = rng.randint(0, min(12, max_edges))

    # curfews chosen in a small range so the strict boundary actually bites
    # (arrivals landing exactly on a curfew are common with these numbers).
    c = [rng.randint(1, 12) for _ in range(n)]

    edges = []
    for _ in range(m):
        u = rng.randint(1, n)
        v = rng.randint(1, n)
        if u == v:
            continue                     # no self loops (irrelevant to answer)
        w = rng.randint(1, 6)
        edges.append((u, v, w))

    out = []
    out.append(f"{n} {len(edges)}")
    out.append(" ".join(str(x) for x in c))
    for (u, v, w) in edges:
        out.append(f"{u} {v} {w}")
    sys.stdout.write("\n".join(out) + "\n")

main()
