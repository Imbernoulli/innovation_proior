# TIER: invalid
# Emits a negative coupling (physically impossible) and the wrong count -> Ratio 0.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    # negative values + too few numbers: guaranteed feasibility violation
    print(" ".join(["-1"] * (n - 2)))


if __name__ == "__main__":
    main()
