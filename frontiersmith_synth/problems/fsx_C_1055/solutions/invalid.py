# TIER: invalid
# Emits the right token count but aims every shot at column L (one past the
# valid range [0, L-1]) -- always rejected by feasibility.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it)); R = int(next(it)); M = int(next(it)); T = int(next(it))
    print(" ".join([str(L)] * T))


if __name__ == "__main__":
    main()
