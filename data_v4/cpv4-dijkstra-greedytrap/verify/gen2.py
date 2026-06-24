import sys
import random

# Harder generator: denser graphs, more colors, larger S to make surcharges bite,
# higher chance node n is reachable, occasional S=0, occasional disconnected.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(2, 7)
    C = rng.randint(1, 4)
    S = rng.choice([0, 0, 1, 3, 5, 10, 20])
    max_m = min(20, n * n + 4)
    m = rng.randint(1, max_m)
    lines = [f"{n} {m} {S}"]
    for _ in range(m):
        u = rng.randint(1, n)
        v = rng.randint(1, n)
        c = rng.randint(1, C)
        w = rng.choice([0, 0, 1, 2, 3, 5, 9, 12])
        lines.append(f"{u} {v} {c} {w}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
