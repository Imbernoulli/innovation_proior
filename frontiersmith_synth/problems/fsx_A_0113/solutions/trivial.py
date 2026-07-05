# TIER: trivial
"""Uniform grid: place all n substations on a ceil(sqrt(n)) x ceil(sqrt(n)) grid,
every coverage disk with the same radius 1/(2k). This reproduces the checker's
own baseline construction -> Ratio ~ 0.1. Ignores the weights entirely."""
import sys, math


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    k = int(math.ceil(math.sqrt(n)))
    r = 1.0 / (2.0 * k)
    out = []
    for i in range(n):
        row = i // k
        col = i % k
        cx = (col + 0.5) / k
        cy = (row + 0.5) / k
        out.append("%.9f %.9f %.9f" % (cx, cy, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
