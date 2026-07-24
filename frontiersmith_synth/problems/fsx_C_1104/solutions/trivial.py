# TIER: trivial
# All phase offsets zero: every job starts its runs at instant 0.
# Reproduces the checker's own baseline B exactly -> ratio ~0.1.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    n = int(next(it))
    for _ in range(n):
        next(it)
        next(it)
    print(" ".join(["0"] * n))


if __name__ == "__main__":
    main()
