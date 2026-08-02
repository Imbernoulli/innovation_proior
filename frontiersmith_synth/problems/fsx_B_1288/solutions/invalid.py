# TIER: invalid
import sys


def main():
    sys.stdin.read()
    # negative velocity threshold, zero window, negative amount cap: all three
    # feasibility checks fail at once.
    print("-5 0 -12.5")


if __name__ == "__main__":
    main()
