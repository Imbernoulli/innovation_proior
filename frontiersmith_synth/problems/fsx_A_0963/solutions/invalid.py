# TIER: invalid
"""Emits a schedule using a disallowed operator (power) -- must be rejected (Ratio 0)."""
import sys


def main():
    sys.stdin.read()
    print("0.5*z**2 + 999999999")


if __name__ == "__main__":
    main()
