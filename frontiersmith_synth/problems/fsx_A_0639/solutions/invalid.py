# TIER: invalid
import sys


def main():
    d = sys.stdin.read().split()
    n = int(d[0])
    # garbage: not a permutation (all zeros) -> must be rejected by the checker
    print(" ".join("0" for _ in range(n)))


if __name__ == "__main__":
    main()
