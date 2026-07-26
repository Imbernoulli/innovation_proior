# TIER: invalid
"""Deliberately infeasible: blow the edit budget by flattening the whole roof
to a huge constant height. Also emits the right token count so the checker's
shape/parsing checks pass and the budget/slope checks are what catch it."""
import sys


def main():
    data = sys.stdin.read().split('\n')
    idx = 0
    R, C = map(int, data[idx].split()); idx += 1
    out = []
    for _ in range(R):
        out.append(' '.join(['999999'] * C))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()
