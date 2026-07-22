# TIER: invalid
# Emits an infeasible artifact: claims more edges than the budget and uses a
# non-finite conductance -> the checker must score this 0.
import sys


def main():
    sys.stdin.read()
    print(10 ** 9)
    print("0 1 nan")


if __name__ == "__main__":
    main()
