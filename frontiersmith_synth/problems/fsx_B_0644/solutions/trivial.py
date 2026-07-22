# TIER: trivial
"""Spend nothing: every edge stays at its base failure probability.
This reproduces the checker's own zero-upgrade baseline construction."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    m = int(toks[idx]); idx += 1
    k = int(toks[idx]); idx += 1
    B = int(toks[idx]); idx += 1
    idx += k  # terminals, unused
    idx += 5 * m  # edges, unused
    print("\n".join(["0"] * m))


if __name__ == "__main__":
    main()
