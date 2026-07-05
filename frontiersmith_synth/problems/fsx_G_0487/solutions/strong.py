# TIER: strong
# Multi-restart randomized greedy: run the corner-avoiding greedy under many seeded random
# scan orders and keep the largest schedule found, using the deterministic lexicographic pass
# as one guaranteed candidate. Randomized restarts routinely beat any single fixed order.
import sys
import random


def read_instance():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); nb = int(next(it))
    blocked = set()
    for _ in range(nb):
        r = int(next(it)); c = int(next(it))
        blocked.add((r, c))
    return N, blocked


def creates_corner(rowmap, colmap, r, c):
    row = rowmap.get(r)
    col = colmap.get(c)
    if row and col:
        for cc in row:
            d = cc - c
            if d >= 1 and (r + d) in col:
                return True
    if row:
        for cc in row:
            d = c - cc
            if d >= 1:
                colcc = colmap.get(cc)
                if colcc and (r + d) in colcc:
                    return True
    if col:
        for rr in col:
            d = r - rr
            if d >= 1:
                rowrr = rowmap.get(rr)
                if rowrr and (c + d) in rowrr:
                    return True
    return False


def greedy_pass(blocked, order):
    rowmap = {}
    colmap = {}
    S = []
    for (r, c) in order:
        if (r, c) in blocked:
            continue
        if creates_corner(rowmap, colmap, r, c):
            continue
        S.append((r, c))
        rowmap.setdefault(r, set()).add(c)
        colmap.setdefault(c, set()).add(r)
    return S


def main():
    N, blocked = read_instance()
    base = [(r, c) for r in range(N) for c in range(N)]

    best = greedy_pass(blocked, base)          # deterministic lexicographic candidate
    for s in range(24):
        rng = random.Random(1000 + s * 7 + N)  # fully deterministic seeds
        order = base[:]
        rng.shuffle(order)
        cand = greedy_pass(blocked, order)
        if len(cand) > len(best):
            best = cand

    out = [str(len(best))]
    for (r, c) in best:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


main()
