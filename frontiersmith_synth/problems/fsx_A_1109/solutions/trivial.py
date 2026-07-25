# TIER: trivial
# Do-nothing schedule: zero pivots.  Feasible; leaves the crystal untouched.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    print(0)


if __name__ == "__main__":
    main()
