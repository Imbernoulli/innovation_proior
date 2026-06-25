import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(1, 9)
    L = rng.randint(1, n)
    # values: mix of negative/positive, sometimes large to stress overflow paths
    big = rng.random() < 0.4
    vals = []
    for _ in range(n):
        if big:
            vals.append(rng.randint(-10**9, 10**9))
        else:
            vals.append(rng.randint(-10, 10))
    print(n, L)
    print(' '.join(map(str, vals)))

main()
