# TIER: invalid
# Deliberately infeasible artifact: a negative allocation entry, which must be
# rejected by the checker's strict feasibility gate regardless of the instance.
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0]); G = int(data[1])

    rows = []
    for i in range(N):
        row = [0] * G
        if i == 0:
            row[0] = -1
        rows.append(" ".join(map(str, row)))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
