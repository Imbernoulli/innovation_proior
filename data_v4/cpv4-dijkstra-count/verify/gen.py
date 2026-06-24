import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 8)
    # Allow many parallel / equal-weight edges to stress the counting/dedup pitfall.
    m = rng.randint(0, 18)
    s = rng.randint(0, n - 1)

    # Small STRICTLY POSITIVE weights => lots of equal-distance ties and many
    # parallel shortest routes, which is exactly where double-count / off-by-one
    # bugs surface. Positive weights keep the shortest-path DAG acyclic, so the
    # number of shortest routes is always finite.
    edges = []
    for _ in range(m):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        w = rng.randint(1, 4)
        edges.append((u, v, w))

    lines = [f"{n} {m} {s}"]
    for (u, v, w) in edges:
        lines.append(f"{u} {v} {w}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
