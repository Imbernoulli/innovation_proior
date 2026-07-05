# TIER: greedy
# Deterministic lexicographic single-pass greedy: scan cells in row-major order and reserve
# each free cell iff it completes no conflict corner with the cells reserved so far.
import sys


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
    row = rowmap.get(r)          # cols present in row r
    col = colmap.get(c)          # rows present in col c
    # role: (r,c) is the corner vertex -> need (r+d,c) and (r,c+d)
    if row and col:
        for cc in row:
            d = cc - c
            if d >= 1 and (r + d) in col:
                return True
    # role: (r,c) is the right end -> vertex (r,c-d), top (r+d,c-d)
    if row:
        for cc in row:
            d = c - cc
            if d >= 1:
                colcc = colmap.get(cc)
                if colcc and (r + d) in colcc:
                    return True
    # role: (r,c) is the top end -> vertex (r-d,c), right (r-d,c+d)
    if col:
        for rr in col:
            d = r - rr
            if d >= 1:
                rowrr = rowmap.get(rr)
                if rowrr and (c + d) in rowrr:
                    return True
    return False


def solve(N, blocked, order):
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
    order = [(r, c) for r in range(N) for c in range(N)]
    S = solve(N, blocked, order)
    out = [str(len(S))]
    for (r, c) in S:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


main()
