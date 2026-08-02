# TIER: trivial
"""Baseline construction: invalidate-on-write for every line (classic MSI).
This exactly reproduces the checker's own internal baseline B, so it should
score ~0.1 on every case."""
import sys


def main():
    data = sys.stdin.read().split()
    L = int(data[2])
    print("\n".join(["INV 1 0.0"] * L))


if __name__ == "__main__":
    main()
