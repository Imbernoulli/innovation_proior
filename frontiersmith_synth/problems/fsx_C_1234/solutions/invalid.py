# TIER: invalid
"""Emits a garbage artifact: too few tokens and an out-of-range protocol
code. Must be rejected by the checker (Ratio: 0.0)."""
import sys


def main():
    data = sys.stdin.read().split()
    L = int(data[0])
    # Deliberately wrong: one fewer token than required, plus a code (9)
    # outside the valid {0,1,2} range.
    toks = ["9"] * max(1, L - 1)
    print(" ".join(toks))


if __name__ == "__main__":
    main()
