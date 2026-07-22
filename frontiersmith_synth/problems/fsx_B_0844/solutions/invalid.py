# TIER: invalid
"""Deliberately infeasible: wrong token count and non-numeric / negative garbage."""
import sys


def main():
    sys.stdin.read()
    print("-1 not_a_number 3.14 nan")


if __name__ == "__main__":
    main()
