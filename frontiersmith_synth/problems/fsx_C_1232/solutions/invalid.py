# TIER: invalid
"""
Deliberately infeasible: references a resource-line index that is out of
the declared [0, M) range (a garbage/huge handle), which the checker must
reject outright with Ratio 0.0.
"""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); C = int(next(it)); budget = int(next(it))

    M = 3
    lines = [
        "C 0",
        "O 999999999 0",   # r way out of [0, M) -> format violation
        "C 0",
    ]
    out = [str(M)] + lines
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
