# TIER: invalid
# Emits an infeasible artifact (vent index 0 is out of range) -> must score 0.
import sys


def main():
    sys.stdout.write("3\n0 -1 9999\n")


main()
