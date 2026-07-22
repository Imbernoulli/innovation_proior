# TIER: invalid
# Emits a garbage priority expression that evaluates to a non-finite value
# (division by zero) -- must be rejected with Ratio: 0.0.
import sys


def main():
    sys.stdin.read()
    print("PRIORITY q / (a - a)")


if __name__ == "__main__":
    main()
