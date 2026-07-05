# TIER: invalid
"""Emits an expression that references a disallowed identifier, so the grader's
whitelist rejects it as infeasible -> Ratio 0.0."""
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("reactor_load + 3.0*mystery_channel\n")


if __name__ == "__main__":
    main()
