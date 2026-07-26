# TIER: invalid
# Garbage: declares the sheet fully unfolded without ever punching any hole,
# so the produced hole set is empty and (for every non-degenerate instance)
# mismatches the target -- must score 0.
import sys


def main():
    sys.stdin.read()
    print("UNFOLD_ALL")


if __name__ == "__main__":
    main()
