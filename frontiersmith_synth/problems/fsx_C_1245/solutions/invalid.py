# TIER: invalid
"""Claims a block of length N+1, which is both out of the checker's valid
per-block length range (> N) and would make the lengths sum to more than N:
rejected unconditionally, on every instance, regardless of content."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); H = int(next(it)); C = int(next(it))
    _A = [int(next(it)) for _ in range(N)]

    print(1)
    print(N + 1, 1)


if __name__ == "__main__":
    main()
