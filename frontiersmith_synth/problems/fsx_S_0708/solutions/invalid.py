# TIER: invalid
import sys


def main():
    sys.stdin.readline()
    # malformed: dangling forward references that no valid SLP could have,
    # plus a bogus tag -- must be rejected by the checker with Ratio: 0.0
    print("T z")
    print("C 5 9")
    print("Q 1")


if __name__ == "__main__":
    main()
