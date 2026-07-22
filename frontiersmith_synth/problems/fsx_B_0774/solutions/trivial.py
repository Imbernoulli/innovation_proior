# TIER: trivial
"""Reproduces the checker's own weak internal baseline: ignore the targets
entirely and just cycle through the K allowed letters. Feasible, but does not
even try to realize any border."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    # lam, alpha, m, target pairs are all ignored by design
    phi = 1.6180339887498949
    W = [int(((i + 1) * phi) % 1.0 * K) for i in range(n)]
    sys.stdout.write(" ".join(map(str, W)) + "\n")


if __name__ == "__main__":
    main()
