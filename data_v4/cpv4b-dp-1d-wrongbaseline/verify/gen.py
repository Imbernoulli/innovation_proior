import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(1, 8)
    # Mix of regimes to exercise the all-negative / mixed / all-positive corners.
    mode = rng.randint(0, 4)
    vals = []
    for _ in range(n):
        if mode == 0:
            vals.append(rng.randint(-9, -1))        # all negative
        elif mode == 1:
            vals.append(rng.randint(1, 9))          # all positive
        elif mode == 2:
            vals.append(rng.randint(-9, 9))         # mixed
        elif mode == 3:
            vals.append(rng.randint(-3, 3))         # small mixed, lots of zeros
        else:
            vals.append(rng.choice([-9, -1, 0, 1, 9]))
    print(n)
    print(' '.join(map(str, vals)))

main()
