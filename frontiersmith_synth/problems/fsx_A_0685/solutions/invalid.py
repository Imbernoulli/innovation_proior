# TIER: invalid
# Emits an out-of-range, infeasible artifact (must score 0.0).
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); k = int(data[1])
    M = n // k
    B = [-1] * k
    T = [-1] * M
    print(" ".join(map(str, B)))
    print(" ".join(map(str, T)))


if __name__ == "__main__":
    main()
