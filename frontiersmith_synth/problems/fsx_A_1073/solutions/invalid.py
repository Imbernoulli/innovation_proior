# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    # Emit a single probe with an out-of-range (negative) position: must score Ratio: 0.0
    # under the feasibility gate regardless of instance size.
    print("1")
    print("-5 F")


main()
