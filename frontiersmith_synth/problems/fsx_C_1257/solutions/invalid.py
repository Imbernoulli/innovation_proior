# TIER: invalid
"""
Emits an out-of-schema policy: P0=0 (a sampling period must be >= 1 to mean anything -- period
0 is not a valid cadence). Deterministically rejected by the checker's range check on P0.
"""
import sys


def main():
    sys.stdin.read()
    print("0 0 0 0")


if __name__ == "__main__":
    main()
