# TIER: invalid
# Emits k matrices with determinant 2 (mod p), not 1 -- infeasible.
import sys


def main():
    data = sys.stdin.read().split("\n")
    p, k, r = map(int, data[0].split())
    for _ in range(k):
        print(f"2 0 0 1")  # det = 2*1 - 0*0 = 2 != 1 (mod p)


if __name__ == "__main__":
    main()
