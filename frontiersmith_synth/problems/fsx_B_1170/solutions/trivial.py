# TIER: trivial
"""Ignores the data entirely: guess K sources at the fixed reference cell
(1,1) with onset time 0. Reproduces the checker's own baseline B."""
import sys


def main():
    toks = sys.stdin.read().split()
    N, T, K = int(toks[0]), int(toks[1]), int(toks[2])
    out = []
    for _ in range(K):
        out.append("1 1 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
