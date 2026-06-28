import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # small cases for differential testing
    n = rng.randint(1, 60)
    q = rng.randint(1, 80)
    # value bound: keep small-ish but exercise multiple bits
    VMAX = (1 << rng.randint(1, 30)) - 1

    lines = []
    lines.append(f"{n} {q}")
    vals = [rng.randint(0, VMAX) for _ in range(n)]
    lines.append(" ".join(map(str, vals)))

    # build a random tree on nodes 1..n
    edges = []
    for v in range(2, n + 1):
        u = rng.randint(1, v - 1)   # random parent among earlier nodes
        edges.append((u, v))
    # optionally make it more path-like / star-like sometimes to stress HLD chains
    shape = rng.random()
    if shape < 0.2 and n >= 2:
        # path
        edges = [(v - 1, v) for v in range(2, n + 1)]
    elif shape < 0.35 and n >= 2:
        # star
        edges = [(1, v) for v in range(2, n + 1)]
    for (u, v) in edges:
        lines.append(f"{u} {v}")

    for _ in range(q):
        t = rng.randint(1, 2)
        u = rng.randint(1, n)
        v = rng.randint(1, n)
        if t == 1:
            x = rng.randint(0, VMAX)
            lines.append(f"1 {u} {v} {x}")
        else:
            lines.append(f"2 {u} {v}")

    sys.stdout.write("\n".join(lines) + "\n")

main()
