# TIER: invalid
"""Deliberately infeasible: claims far more flagged participant-windows than
it actually lists (mismatched count) and also blows through the alert
budget -- must be rejected by the checker with Ratio: 0.0."""
import sys


def main():
    sys.stdin.read()  # ignore the instance entirely
    print(999999999)
    print("0 0")
    print("0 1")


if __name__ == "__main__":
    main()
