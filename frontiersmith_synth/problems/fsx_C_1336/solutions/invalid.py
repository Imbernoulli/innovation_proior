# TIER: invalid
# Emit an out-of-range coformer index regardless of the instance. Infeasible
# on every case -> checker must score 0.
import sys


def main():
    sys.stdin.read()
    print("-1 -1")


if __name__ == "__main__":
    main()
