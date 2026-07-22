# TIER: invalid
"""Deliberately infeasible: dumps every task onto a single line in raw id order,
ignoring both the horizon cap and precedence entirely (must score 0)."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(data[p]); p += 1
        return v

    N = nxt(); M = nxt(); k = nxt(); H = nxt()
    # ignore the rest
    ids = list(range(1, N + 1))
    print(1)
    print(str(len(ids)) + " " + " ".join(map(str, ids)))


if __name__ == "__main__":
    main()
