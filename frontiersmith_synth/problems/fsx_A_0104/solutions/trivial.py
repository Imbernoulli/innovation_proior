# TIER: trivial
"""Reproduce the checker's baseline: the quadratic-residue point set
P_i = (i/n, (i^2 mod n)/n).  This is exactly what the checker uses as B, so it
scores Ratio = 0.1 on every case."""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = []
    for i in range(n):
        out.append("%.10f %.10f" % (i / n, (i * i % n) / n))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
