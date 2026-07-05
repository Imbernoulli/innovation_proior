# TIER: invalid
"""Emit a deliberately infeasible schedule: assign depot 1 to EVERY cell. This
violates row/column uniqueness (and overwrites prefilled cells), so the checker
must score it 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    out = []
    for i in range(N):
        out.append(' '.join('1' for _ in range(N)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
