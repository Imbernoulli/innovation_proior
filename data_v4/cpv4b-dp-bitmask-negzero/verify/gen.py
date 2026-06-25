import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rnd = random.Random(seed)

    n = rnd.randint(0, 6)
    # Bias toward distributions that exercise the negative/zero corners:
    #  - "all-negative" matrices (answer must be 0 via empty roster)
    #  - matrices heavy in zeros
    #  - mixed
    mode = rnd.randint(0, 4)
    if mode == 0:           # all strictly negative
        lo, hi = -20, -1
    elif mode == 1:         # negatives and zeros, no positives
        lo, hi = -10, 0
    elif mode == 2:         # zeros and a few positives
        lo, hi = 0, 5
    elif mode == 3:         # small symmetric range incl. 0
        lo, hi = -6, 6
    else:                   # wider range
        lo, hi = -50, 50

    out = [str(n)]
    for i in range(n):
        row = [str(rnd.randint(lo, hi)) for _ in range(n)]
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")

main()
