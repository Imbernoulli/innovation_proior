# TIER: invalid
"""Deliberately infeasible: references a cell far out of range, rakes a
block whose width isn't a power of two, and never actually reaches the
threshold. Must score 0 on every case."""
import sys


def main():
    sys.stdin.read()  # ignore the instance entirely
    out = [
        "3",
        "P 999999",   # out-of-range point index
        "B 1 3",      # width 3 is not a power of two, and misaligned
        "Q 5 5",      # unknown op code
    ]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
