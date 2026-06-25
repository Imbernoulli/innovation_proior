import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 10)
    # Keep coordinate space small so overlaps and exact-touching (end == start) happen often.
    T = rng.randint(2, 8)
    lines = [str(n)]
    for _ in range(n):
        s = rng.randint(0, T - 1)
        e = rng.randint(s + 1, T)          # ensure s < e (non-empty interval)
        v = rng.randint(0, 12)             # non-negative values
        lines.append(f"{s} {e} {v}")
    sys.stdout.write("\n".join(lines) + "\n")

main()
