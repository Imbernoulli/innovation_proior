# TIER: trivial
"""Reproduces the checker's own baseline: one slab, no cuts at all."""
import sys


def main():
    data = sys.stdin.read().split()
    # header is enough; we don't need to read the rest
    _L, _Q, _BASE = data[0], data[1], data[2]
    print(0)
    print()


if __name__ == "__main__":
    main()
