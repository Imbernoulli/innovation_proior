# TIER: invalid
import sys


def main():
    # Read and discard the instance, then claim the whole guild is already
    # settled with zero transfers. On every generated instance at least one
    # party has a genuinely nonzero net balance, so this is infeasible and
    # must be rejected by the checker (Ratio: 0.0).
    sys.stdin.read()
    print("0")


if __name__ == "__main__":
    main()
