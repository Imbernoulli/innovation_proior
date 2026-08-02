# TIER: invalid
"""Emits garbage: non-finite tokens where integers are expected. Must score 0."""
import sys


def main():
    sys.stdin.read()  # consume input, ignored
    print("nan nan")
    print("inf -inf")
    print("999999999999999999999999 3")
    print("0 1")


if __name__ == "__main__":
    main()
