import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 9)

    # Build a connected graph on nodes 1..n by first laying down a random spanning tree,
    # then adding a few extra random edges (no self-loops; multi-edges allowed but harmless).
    edges = []
    for v in range(2, n + 1):
        u = rng.randint(1, v - 1)
        edges.append((u, v))

    extra = rng.randint(0, n)  # a handful of extra edges to create cycles / varied distances
    for _ in range(extra):
        u = rng.randint(1, n)
        v = rng.randint(1, n)
        if u != v:
            edges.append((u, v))

    rng.shuffle(edges)
    m = len(edges)

    out = [f"{n} {m}"]
    for (u, v) in edges:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
