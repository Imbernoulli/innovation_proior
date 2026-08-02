# TIER: invalid
"""Emits a garbage artifact: a non-finite rate. Must score 0."""
import sys


def main():
    sys.stdout.write("0 nan\n")


if __name__ == "__main__":
    main()
