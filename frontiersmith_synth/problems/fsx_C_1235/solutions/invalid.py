# TIER: invalid
"""Emits a plan with an out-of-range cutover tick -- must score 0."""
import sys


def main():
    head = sys.stdin.readline().split()
    K, T, M = int(head[0]), int(head[1]), int(head[2])
    sys.stdin.readline()
    # cutover far outside [0, T], plus bogus (non 0/1) flags
    sys.stdout.write(f"{T + 999999}\n{' '.join(['7'] * M)}\n")


if __name__ == "__main__":
    main()
