# TIER: invalid
"""
Deliberately infeasible: claims the top modulation tier for every channel while
launching at 1.5x the declared power ceiling Pmax_c (violates the power-range
constraint) -- must score 0.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    C = int(nx())
    S = int(nx())
    for _ in range(S):
        nx()
    for _ in range(C):
        nx()  # eta
    K = int(nx())
    for _ in range(K):
        nx(); nx()
    pmax = [float(nx()) for _ in range(C)]
    nx()  # baud
    for _ in range(C):
        for _ in range(C):
            nx()  # kappa

    lines = []
    for c in range(C):
        lines.append("%.9f %d" % (pmax[c] * 1.5 + 5.0, K))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
