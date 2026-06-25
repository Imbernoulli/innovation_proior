import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # small cases so the brute (which explores merge orders) stays fast.
    n = rng.randint(0, 8)
    print(n)
    if n > 0:
        vals = [rng.randint(1, 12) for _ in range(n)]
        print(' '.join(map(str, vals)))

if __name__ == "__main__":
    main()
