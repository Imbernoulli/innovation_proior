# TIER: invalid
"""Emits a syntactically-plausible but out-of-range edit (a state id far
outside the automaton), which the checker must reject as infeasible."""
import sys


def main():
    sys.stdin.read()
    print(1)
    print("TOGGLE 999999")


if __name__ == "__main__":
    main()
