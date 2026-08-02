# TIER: greedy
# The "obvious" recipe: count how many OTHER transactions each transaction
# conflicts with (any shared key, read or write, direction ignored) and
# pick an isolation level off a fixed threshold table. This is exactly the
# instinct "few conflicts -> a light isolation level is enough" -- it never
# looks at whether the conflicts it does have close a directed cycle, so a
# ring of low-degree write-skew partners (every member touching only its
# two cycle neighbours) sails through at SNAPSHOT and the cycle survives.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    R = [set() for _ in range(N)]
    W = [set() for _ in range(N)]
    for i in range(N):
        w = int(next(it))
        nr = int(next(it))
        for _ in range(nr):
            R[i].add(int(next(it)))
        nw = int(next(it))
        for _ in range(nw):
            W[i].add(int(next(it)))

    conflicts = [set() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if (R[i] & W[j]) or (W[i] & R[j]) or (W[i] & W[j]):
                conflicts[i].add(j)

    lvl = []
    for i in range(N):
        deg = len(conflicts[i])
        if deg == 0:
            lvl.append(0)
        elif deg <= 2:
            lvl.append(1)
        else:
            lvl.append(2)

    print(" ".join(str(x) for x in lvl))


if __name__ == "__main__":
    main()
