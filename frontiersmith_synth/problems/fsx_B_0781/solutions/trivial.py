# TIER: trivial
"""Constant-zero predictor -- ignores the drive entirely. Reproduces the
checker's own internal baseline (Ratio ~= 0.1)."""
import sys


def main():
    sys.stdin.read()  # consume input, unused
    print("OUT 0")


if __name__ == "__main__":
    main()
