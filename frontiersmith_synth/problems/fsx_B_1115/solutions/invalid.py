# TIER: invalid
import sys


def main():
    sys.stdin.read()  # ignore the instance entirely
    # a single Stage-3 lot drawing a large quantity from an empty Buffer 2,
    # and no demand pulse is ever actually met -> guaranteed infeasible.
    print(1)
    print("3 0 999999.0")


if __name__ == "__main__":
    main()
