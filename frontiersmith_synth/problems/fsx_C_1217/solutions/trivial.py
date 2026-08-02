# TIER: trivial
"""Do nothing: submit zero edits. This reproduces the checker's own baseline
(the damaged automaton's raw accuracy) -- the trivial feasible construction."""
import sys


def main():
    sys.stdin.read()  # consume input (unused)
    print(0)


if __name__ == "__main__":
    main()
