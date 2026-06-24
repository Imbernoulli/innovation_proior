import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 8)
    # allow multi-edges and occasional self-loops to stress dedup; small m to keep brute tractable
    max_m = min(14, n * (n - 1) // 2 + 4)
    m = rng.randint(0, max_m)

    edges = []
    for _ in range(m):
        u = rng.randint(1, n)
        v = rng.randint(1, n)
        edges.append((u, v))

    s = rng.randint(1, n)
    t = rng.randint(1, n)

    out = []
    out.append(f"{n} {m} {s} {t}")
    for (u, v) in edges:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
