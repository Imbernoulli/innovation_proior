import sys, random

def main():
    seed = int(sys.argv[1])
    rnd = random.Random(seed)
    n = rnd.randint(2, 8)
    k = rnd.randint(2, n)
    # bias toward negative / zero coordinates and duplicates to stress sign handling
    lo, hi = rnd.choice([(-6, 6), (-9, 0), (-4, 4), (-9, -1), (-2, 2), (0, 0)])
    x = [rnd.randint(lo, hi) for _ in range(n)]
    print(n, k)
    print(" ".join(map(str, x)))

if __name__ == "__main__":
    main()
