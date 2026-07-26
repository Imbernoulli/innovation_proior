# TIER: invalid
"""Emits an out-of-grammar / non-finite artifact: must score 0."""
import sys


def main():
    sys.stdin.read()
    # t**500 blows up to inf well inside the quadrature domain for the
    # observed window scales used by this family, AND uses a disallowed
    # bare numeric-overflow pattern; also throw in a disallowed name to be
    # doubly sure the checker rejects this on parse.
    print("undefined_symbol + t**500")


if __name__ == "__main__":
    main()
