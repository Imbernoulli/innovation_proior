# TIER: strong
# Halton (base 2,3) low-discrepancy sequence for the added stations -- a genuine 2D
# quasi-random layout that samples the reef far more evenly than the diagonal/centre.
import sys


def halton(i, base):
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def main():
    toks = sys.stdin.read().split()
    d = int(toks[0]); M = int(toks[1]); K = int(toks[2])
    A = M - K
    out = []
    for i in range(1, A + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        out.append("%.6f %.6f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
