import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(2, 6)
    # small number of colors so transfers actually matter
    C = rng.randint(1, 3)
    S = rng.randint(0, 8)
    # edges: allow multi-edges and self-loops (self-loops just add cost, never help)
    max_m = min(14, n * n)
    m = rng.randint(1, max_m)
    lines = [f"{n} {m} {S}"]
    for _ in range(m):
        u = rng.randint(1, n)
        v = rng.randint(1, n)
        c = rng.randint(1, C)
        w = rng.randint(0, 9)
        lines.append(f"{u} {v} {c} {w}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
