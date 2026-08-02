# TIER: invalid
"""Emit a deliberately infeasible design: area fractions that do not sum to
1 (and a radius outside the allowed manufacturing range). The checker must
reject this with Ratio: 0.0."""
import sys


def main():
    sys.stdin.read()  # consume the instance, ignore it
    print(1)
    print(0.0)
    print("%.6f %.6f" % (1e9, 3.7))  # radius out of range, weight != 1


if __name__ == "__main__":
    main()
