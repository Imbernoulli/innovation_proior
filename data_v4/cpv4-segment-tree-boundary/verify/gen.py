import sys, random

def main():
    seed = int(sys.argv[1])
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    q = rng.randint(1, 18)
    # Small value alphabet so ties / equal-neighbour boundaries are common;
    # that is exactly where the strict-ascent seam logic gets stressed.
    VMAX = rng.choice([1, 2, 3, 5, 9])
    print(n, q)
    print(" ".join(str(rng.randint(0, VMAX)) for _ in range(n)))
    for _ in range(q):
        if rng.randint(1, 2) == 1:
            p = rng.randint(1, n)
            x = rng.randint(0, VMAX)
            print(1, p, x)
        else:
            l = rng.randint(1, n)
            r = rng.randint(l, n)
            print(2, l, r)

main()

