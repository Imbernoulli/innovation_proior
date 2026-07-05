# TIER: greedy
# Place added stations along the reef's main diagonal -- covers each axis but leaves
# the off-diagonal reef flats undersampled (a mediocre 1D-style layout).
import sys


def main():
    toks = sys.stdin.read().split()
    d = int(toks[0]); M = int(toks[1]); K = int(toks[2])
    A = M - K
    out = []
    for i in range(A):
        t = (i + 0.5) / A
        out.append("%.6f %.6f" % (t, t))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
