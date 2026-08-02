# TIER: trivial
"""
Never checkpoint. On every failure, redo everything from the very start. This is the
textbook "do nothing" baseline: cheap to write, catastrophic once failures cluster.
"""
import sys


def main():
    sys.stdin.read()  # instance is unused
    print(0)
    print()


if __name__ == "__main__":
    main()
