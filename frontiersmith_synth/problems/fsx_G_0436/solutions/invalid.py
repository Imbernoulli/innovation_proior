# TIER: invalid
"""Emit a disallowed / non-parsing expression (references an unknown identifier
and uses no valid form) -> infeasible -> score 0."""
import sys


def main():
    _ = sys.stdin.read()
    print("bandwidth ** +* / rho")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
