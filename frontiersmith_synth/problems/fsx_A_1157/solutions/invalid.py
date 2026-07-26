# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    # wildly over budget -- infeasible on any instance
    print(" ".join(["777777777"] * T))


if __name__ == "__main__":
    main()
