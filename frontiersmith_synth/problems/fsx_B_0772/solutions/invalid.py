# TIER: invalid
"""Garbage artifact: a single region equal to the whole bounding grid.  It swallows empty
(non-solid) cells the real sculpture doesn't occupy, so the checker's coverage gate must
reject it."""
import sys


def main():
    data = sys.stdin.read().split()
    X, Y, Z = int(data[0]), int(data[1]), int(data[2])
    print(1)
    print("0 0 0 %d %d %d 0" % (X, Y, Z))


if __name__ == "__main__":
    main()
