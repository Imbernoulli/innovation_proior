# TIER: invalid
"""Claims a resolution/rate/order/gain combo whose cost blows the budget by
a large margin (R set far beyond what the budget allows): rejected
unconditionally, on every instance, regardless of content."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    NBINS = int(next(it)); BUDGET = int(next(it)); WFILT = int(next(it))
    _ = [next(it) for _ in range(3)]
    for _ in range(NBINS):
        next(it); next(it)

    print(16, BUDGET + 1000, 8, 12)


if __name__ == "__main__":
    main()
