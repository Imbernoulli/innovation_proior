# TIER: invalid
"""Claims every item at once -- guaranteed to blow at least one Scheme-A class
quota (classes are always sized below n by construction), so the checker must
reject it with Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    idxs = list(range(1, n + 1))
    print(n)
    print(" ".join(map(str, idxs)))
    print(1)
    print(str(n) + " " + " ".join(map(str, idxs)))


if __name__ == "__main__":
    main()
