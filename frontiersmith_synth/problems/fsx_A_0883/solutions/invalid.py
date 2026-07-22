# TIER: invalid
# Emits a non-permutation (symbol 0 repeated) -- must be rejected with Ratio 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    k = int(data[1])
    order = [0] * k  # not a permutation
    print(" ".join(map(str, order)))
    print(0)


if __name__ == "__main__":
    main()
