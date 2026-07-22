# TIER: invalid
# Skips sample 1 entirely and tastes sample 2 twice instead -- an infeasible
# flight order (every sample must appear exactly once).
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    order = [2] + list(range(2, N + 1))
    print(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
