import random
import sys

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    n = random.randint(1, 8)
    # Allow a sparse-to-dense range of edges; permit disconnected graphs.
    max_edges = n * (n - 1) // 2
    m = random.randint(0, min(max_edges + 2, 12))

    # Weights are non-negative per the contract (0 <= w <= 1e6). Use a mix of
    # small and large weights; large ones make the int-overflow real even at
    # small n, while brute (Python big ints) stays exact for the equivalence
    # check.
    def rand_w():
        if random.random() < 0.5:
            return random.randint(0, 5)
        return random.randint(0, 1_000_000)
    w = [rand_w() for _ in range(n)]

    edges = []
    for _ in range(m):
        a = random.randint(1, n)
        b = random.randint(1, n)
        edges.append((a, b))  # self-loops and multi-edges allowed; harmless

    out = []
    out.append(f"{n} {len(edges)}")
    out.append(" ".join(str(x) for x in w))
    for (a, b) in edges:
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")

main()
