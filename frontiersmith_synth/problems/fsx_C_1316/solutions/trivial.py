# TIER: trivial
"""Baseline construction: leave every position unsubstituted (all H). This
exactly reproduces the checker's own internal reference baseline (the "do
nothing" pattern), so it should score ~0.1 on every case."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    print(" ".join("0" for _ in range(N)))


if __name__ == "__main__":
    main()
