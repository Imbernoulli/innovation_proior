# TIER: invalid
# Removes only the first waste cell and stops -- the job is never finished,
# so the checker's "all waste must be removed" feasibility gate rejects it
# (Ratio: 0.0).
import sys


def main():
    data = sys.stdin.read().split("\n")
    head = data[0].split()
    R, C, K = int(head[0]), int(head[1]), int(head[2])
    grid = data[1:1+R]

    first = None
    for r in range(R):
        row = grid[r]
        for c in range(C):
            if row[c] == '#':
                first = (r, c)
                break
        if first:
            break

    if first is None:
        print(0)
    else:
        print(1)
        print("R %d %d" % (first[0], first[1]))


main()
