# TIER: trivial
# Baseline construction: put every fitting on the single vertical manifold
# x = 1/2, evenly spaced in y. This reproduces the checker's internal
# baseline exactly, so it scores Ratio ~ 0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    d, M = int(toks[0]), int(toks[1])
    out = []
    for i in range(M):
        out.append("%.10f %.10f" % (0.5, (i + 0.5) / M))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
