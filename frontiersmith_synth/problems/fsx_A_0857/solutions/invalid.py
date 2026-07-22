# TIER: invalid
# Emits an expression using a disallowed name (attribute access) -> Ratio 0.0.
import sys


def main():
    sys.stdin.read()
    print("x0.__class__ + x1 + x2")


if __name__ == "__main__":
    main()
