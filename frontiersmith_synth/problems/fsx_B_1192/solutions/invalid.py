# TIER: invalid
"""Emits an expression that references a disallowed identifier (an undefined
sensor name), so the grader's whitelist rejects it as infeasible -> Ratio 0.0."""
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("x4 * knee_cycle_estimate - 0.5*x2\n")


if __name__ == "__main__":
    main()
