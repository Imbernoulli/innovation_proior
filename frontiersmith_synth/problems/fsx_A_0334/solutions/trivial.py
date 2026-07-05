# TIER: trivial
"""Reproduces the checker's diagonal baseline construction -> scores ~0.1.
Places the n free stations equally spaced along the gallery's main diagonal."""
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    out = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        x = 0.05 + 0.90 * t
        y = 0.05 + 0.90 * t
        out.append("%.6f %.6f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
