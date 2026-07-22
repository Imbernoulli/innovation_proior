# TIER: invalid
import sys


def main():
    sys.stdin.read()
    # syntactically valid expression that overflows to +inf for any a>0;
    # the checker must reject this via its finiteness check.
    print("a * 1e400")


if __name__ == "__main__":
    main()
