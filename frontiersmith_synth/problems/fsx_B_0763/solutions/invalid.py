# TIER: invalid
"""Deliberately infeasible: assigns EVERY regular to (choice1, slot1), guaranteeing
massive cot collisions whenever any room is shared by more than one regular."""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    # ignore the rest; just emit n copies of code 1 (choice1, slot1)
    out = ["1"] * n
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
