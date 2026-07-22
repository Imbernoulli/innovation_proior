# TIER: invalid
"""Deliberately infeasible: drops one errand id (wrong token count) and
reverses the rest (breaks nearly every dependence edge too)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    next(it)  # K

    ids = []
    for _ in range(N):
        oid = int(next(it))
        k = int(next(it))
        for _ in range(k):
            next(it)
        next(it)
        ids.append(oid)

    out = list(reversed(ids))[:-1]  # wrong length AND (generally) wrong order
    sys.stdout.write(" ".join(map(str, out)))


if __name__ == "__main__":
    main()
