import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(2, 8)
    # pick a target edge count; allow parallel edges and self-loops to stress dedup
    max_simple = n * (n - 1) // 2
    m = random.randint(0, max_simple + 3)

    edges = []
    for _ in range(m):
        u = random.randint(1, n)
        v = random.randint(1, n)
        # with some probability force a self-loop or a duplicate of an existing edge
        r = random.random()
        if r < 0.12:
            v = u                       # self-loop
        elif r < 0.30 and edges:
            u, v = random.choice(edges) # duplicate (parallel) edge
        edges.append((u, v))

    lines = [f"{n} {m}"]
    for (u, v) in edges:
        lines.append(f"{u} {v}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
