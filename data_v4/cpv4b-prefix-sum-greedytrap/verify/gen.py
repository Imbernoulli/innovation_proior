import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rnd = random.Random(seed)

    # Mix of regimes to stress the greedy trap and the corners.
    mode = seed % 6
    if mode == 0:
        n = rnd.randint(0, 2)            # tiny / empty
    elif mode == 1:
        n = rnd.randint(1, 8)
    else:
        n = rnd.randint(1, 12)

    lo, hi = -4, 4
    if mode == 3:
        lo, hi = -9, 9                   # wider values, more ties broken
    if mode == 4:
        lo, hi = 0, 5                    # mostly non-negative
    if mode == 5:
        lo, hi = -5, 0                   # mostly non-positive

    a = [rnd.randint(lo, hi) for _ in range(n)]

    out = [str(n)]
    out.extend(str(x) for x in a)
    sys.stdout.write(" ".join(out) + "\n")

main()
