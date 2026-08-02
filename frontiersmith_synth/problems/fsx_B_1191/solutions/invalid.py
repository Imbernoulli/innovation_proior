# TIER: invalid
"""Emits a grammatically-invalid expression (references an undeclared
variable) -- must score 0 under strict feasibility checking."""
import sys


def main():
    sys.stdin.read()
    print("P_measured + 1")


if __name__ == "__main__":
    main()
