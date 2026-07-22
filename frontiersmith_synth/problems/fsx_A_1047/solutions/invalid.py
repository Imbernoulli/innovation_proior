# TIER: invalid
# Garbage artifact: claims a single checkpoint at an out-of-range id, which
# the checker must reject with Ratio: 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    print(1)
    print(n + 999999)


if __name__ == "__main__":
    main()
