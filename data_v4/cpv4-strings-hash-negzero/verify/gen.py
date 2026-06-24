import sys, random

def main():
    seed = int(sys.argv[1])
    rng = random.Random(seed)

    mode = rng.randint(0, 9)
    if mode == 0:
        n = 0
    elif mode == 1:
        n = 1
    else:
        n = rng.randint(2, 12)

    # value distribution chosen to stress negatives, zeros, all-negative
    vmode = rng.randint(0, 6)
    if vmode == 0:        # all negative
        lo, hi = -5, -1
    elif vmode == 1:      # zeros and negatives only
        lo, hi = -3, 0
    elif vmode == 2:      # tiny alphabet (forces repeats)
        lo, hi = -1, 1
    elif vmode == 3:      # zeros heavy
        lo, hi = 0, 2
    elif vmode == 4:      # full small mix
        lo, hi = -4, 4
    elif vmode == 5:      # large magnitude (sign / offset stress)
        lo, hi = -10**9, 10**9
    else:                 # mostly a single repeated value
        lo, hi = -2, 2

    a = [rng.randint(lo, hi) for _ in range(n)]

    out = [str(n)]
    out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")

main()
