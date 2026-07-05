# TIER: invalid
"""Emits an expression that references a disallowed identifier (not the variable r),
so the grader's whitelist rejects it as infeasible -> Ratio 0.0."""
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("mass_density / radius**2 + halo_offset\n")


if __name__ == "__main__":
    main()
