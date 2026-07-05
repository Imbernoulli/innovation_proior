# TIER: invalid
# Emits an expression that overflows to +inf on the held-out (large K,L) points.
# The grader rejects non-finite results -> Ratio 0.0.
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("exp(700.0 * K * L)\n")


if __name__ == "__main__":
    main()
