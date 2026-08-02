# TIER: invalid
"""Deterministic garbage: tries to enable flag 1 in every window, which is
illegal from the second window on (a flag can never be re-touched once it
has been enabled). Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    out = ["E 1" for _ in range(N)]
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
