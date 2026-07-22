# TIER: invalid
"""Emit a program that computes a single doubling and stops: the targets are
never produced, so the equivalence check fails -> Ratio 0.0."""
import sys


def main():
    sys.stdout.write("1\n0 0\n")


if __name__ == "__main__":
    main()
