# TIER: invalid
"""Deliberately infeasible: gives every crate the same offset 0, guaranteed to
overlap for any pair of crates whose stays intersect in time (always exists
for N >= 2 here since intervals are drawn from a shared shuffled timeline)."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def next_int():
        nonlocal pos
        v = int(data[pos])
        pos += 1
        return v

    N = next_int()
    M = next_int()
    next_int()  # PAGE
    next_int()  # LAMBDA
    for _ in range(N):
        next_int()
        next_int()
        next_int()
    for _ in range(M):
        next_int()
        next_int()

    sys.stdout.write("\n".join("0" for _ in range(N)) + "\n")


if __name__ == "__main__":
    main()
