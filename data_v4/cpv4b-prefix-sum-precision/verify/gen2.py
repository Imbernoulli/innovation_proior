import sys, random

# harder generator: larger n, more often big values, sometimes all extreme.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(1, 40)
    L = rng.randint(1, n)
    mode = rng.randint(0, 3)
    vals = []
    for _ in range(n):
        if mode == 0:
            vals.append(rng.randint(-10**9, 10**9))
        elif mode == 1:
            vals.append(rng.choice([-10**9, 10**9, 0]))
        elif mode == 2:
            vals.append(rng.randint(-3, 3))
        else:
            # smoothly increasing/decreasing to build long hulls
            vals.append(rng.randint(-10**9, 10**9))
    print(n, L)
    print(' '.join(map(str, vals)))

main()
