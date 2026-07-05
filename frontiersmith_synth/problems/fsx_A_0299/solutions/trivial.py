# TIER: trivial
"""Reproduces the checker's internal pseudo-random scatter baseline exactly,
so it scores ~0.1. Same LCG, seeded only by (d, M)."""
import sys


def main():
    data = sys.stdin.read().split()
    d, M = int(data[0]), int(data[1])
    seed = (d * 1000003 + M * 97 + 12345) & 0x7FFFFFFF
    s = seed
    out = []
    for _ in range(M):
        coords = []
        for _k in range(d):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            coords.append("%.10f" % (s / 0x7FFFFFFF))
        out.append(" ".join(coords))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
