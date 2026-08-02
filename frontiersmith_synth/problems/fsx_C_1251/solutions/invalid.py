# TIER: invalid
"""Deliberately infeasible: claims to select every candidate id 0..C-1 regardless of
the encoding-space budget K or the area budget A, which is essentially always over both
budgets and must score 0."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    K, A = map(int, data[idx].split()); idx += 1
    C = int(data[idx]); idx += 1

    ids = list(range(C))
    out = [str(len(ids)), " ".join(map(str, ids))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
