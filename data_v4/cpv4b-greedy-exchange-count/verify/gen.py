import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Keep m small so the bitmask brute (<= 2^m states) stays fast.
    m = rng.randint(1, 10)
    n = rng.randint(1, 8)
    lines = [f"{m} {n}"]
    for _ in range(n):
        s = rng.randint(0, m - 1)
        L = rng.randint(1, m)   # contract: 1 <= L <= m
        lines.append(f"{s} {L}")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
