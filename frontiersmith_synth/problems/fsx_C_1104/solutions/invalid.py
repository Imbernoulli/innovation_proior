# TIER: invalid
# Emits offsets that are out of range for every job (offsets must be < p_i).
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    n = int(next(it))
    for _ in range(n):
        next(it)
        next(it)
    print(" ".join(["999983"] * n))


if __name__ == "__main__":
    main()
