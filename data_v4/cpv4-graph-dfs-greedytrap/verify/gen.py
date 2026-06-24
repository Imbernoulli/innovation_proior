import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 9)
    # Build a DAG by fixing a random topological order (0..n-1) and only
    # adding edges from lower index to higher index.
    perm = list(range(n))
    rng.shuffle(perm)
    pos = {node: i for i, node in enumerate(perm)}

    # prestige values: small range, include negatives to exercise stop-option.
    p = [rng.randint(-6, 6) for _ in range(n)]

    edges = []
    # candidate edges respecting topo order
    cand = []
    for a in range(n):
        for b in range(n):
            if pos[a] < pos[b]:
                cand.append((a, b))
    rng.shuffle(cand)
    maxe = rng.randint(0, len(cand))
    for i in range(maxe):
        edges.append(cand[i])

    out = []
    out.append(f"{n} {len(edges)}")
    out.append(" ".join(str(x) for x in p))
    for (u, v) in edges:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")

main()
