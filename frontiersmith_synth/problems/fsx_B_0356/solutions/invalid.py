# TIER: invalid
"""Emits a guaranteed-infeasible instruction (out-of-range SWAP). Scores 0."""
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("S -1 999999\n")


if __name__ == "__main__":
    main()
