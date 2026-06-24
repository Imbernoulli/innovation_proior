import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    # Tiny cases so brute (Bellman-Ford in Python) is fast and obviously correct.
    n = random.randint(1, 7)
    # Allow up to a dense-ish small graph, including possible multi-edges / self loops.
    max_m = n * (n + 1)  # generous, allows parallel edges
    m = random.randint(0, min(max_m, 12))

    lines = []
    lines.append(f"{n} {m}")
    for _ in range(m):
        u = random.randint(1, n)
        v = random.randint(1, n)
        # Mix small and large weights to exercise overflow-prone accumulation
        # even on tiny graphs (large weights summed across a few edges).
        if random.random() < 0.4:
            w = random.randint(0, 1000000000)
        else:
            w = random.randint(0, 20)
        lines.append(f"{u} {v} {w}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
