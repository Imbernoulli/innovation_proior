import sys, random

# Heavier stress: more lanterns, more overlap, slightly larger k.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    m = rng.randint(1, 6)
    pool = [2, 3, 4, 6, 8, 9, 12]   # lots of shared factors -> heavy overlap
    p = [rng.choice(pool) for _ in range(m)]
    k = rng.randint(1, 60)
    print(f"{m} {k}")
    print(" ".join(map(str, p)))

if __name__ == "__main__":
    main()
