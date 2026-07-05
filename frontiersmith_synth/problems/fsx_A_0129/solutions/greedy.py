# TIER: greedy
# A genuine 2-D low-discrepancy construction: the Hammersley point set in
# base 2, x_i = (i+0.5)/M, y_i = radical_inverse_2(i). Much more uniform than
# the vertical-line baseline, so it clearly beats trivial.
import sys


def rev2(i):
    r = 0.0
    f = 0.5
    while i > 0:
        r += (i & 1) * f
        f *= 0.5
        i >>= 1
    return r


def main():
    toks = sys.stdin.read().split()
    d, M = int(toks[0]), int(toks[1])
    out = []
    for i in range(M):
        out.append("%.10f %.10f" % ((i + 0.5) / M, rev2(i)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
