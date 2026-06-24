import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Two regimes selected by the seed so the stress test covers both
    # tiny dense overlaps and slightly larger spread-out cases. Both stay
    # small enough that the O(area) brute force is exact and fast.
    if seed % 3 == 0:
        n = rng.randint(0, 10)
        C = 12
    else:
        n = rng.randint(0, 6)
        C = 8
    lines = [str(n)]
    for _ in range(n):
        a = rng.randint(0, C)
        b = rng.randint(0, C)
        c = rng.randint(0, C)
        d = rng.randint(0, C)
        lines.append(f"{a} {b} {c} {d}")
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
