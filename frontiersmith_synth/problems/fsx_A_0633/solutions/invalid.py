# TIER: invalid
"""Emits a structurally broken touch: claims a long length but the calls
never actually return to rounds (closure violated) and repeat a row along
the way (trueness violated).  Must score 0."""
import sys


def main():
    sys.stdin.read()
    print(5)
    print(1)
    print(1)
    print(1)
    print(1)
    print(1)


if __name__ == "__main__":
    main()
