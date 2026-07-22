# TIER: invalid
# Declares a batch size far above any legal capacity K -- always rejected.
import sys


def main():
    sys.stdin.read()
    print("1")
    print("0 0 999999 1 2 3 4 5 6 7 8 9 10")


if __name__ == "__main__":
    main()
