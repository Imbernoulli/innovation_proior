# TIER: invalid
"""Emit a disallowed expression (uses a conditional / comparison, which is not in
the grader's whitelist) -> infeasible -> Ratio 0.0."""
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("x0 if x0 > 0.5 else x1\n")


if __name__ == "__main__":
    main()
