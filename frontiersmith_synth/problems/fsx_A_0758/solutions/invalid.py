# TIER: invalid
import sys


def main():
    sys.stdin.read()
    print(1)
    print("E 999999999")  # always out of range -> infeasible


if __name__ == "__main__":
    main()
