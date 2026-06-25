import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(0, 60)
    vmax = rng.choice([3, 7, 20, 100])
    L = rng.randint(0, vmax)
    R = rng.randint(L, 2 * vmax)
    f = [rng.randint(-vmax, vmax) for _ in range(n)]
    print(n)
    print(L, R)
    print(" ".join(map(str, f)))

if __name__ == "__main__":
    main()
