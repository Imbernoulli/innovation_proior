# TIER: invalid
# Emits an infeasible schedule: pivot with i == j, an out-of-range index, and a
# trailing garbage token.  Must score 0.
import sys


def main():
    sys.stdout.write("3\n1 1\n2 999999\nx y\n")


if __name__ == "__main__":
    main()
