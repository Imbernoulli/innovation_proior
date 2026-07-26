# TIER: invalid
"""Books a slot id that cannot exist in any instance -> must score 0."""
import sys


def main():
    sys.stdin.read()  # ignore the instance entirely
    print(1)
    print(999999999)


if __name__ == "__main__":
    main()
