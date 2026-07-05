# TIER: greedy
# Lexicographic (row-major) greedy: accept a pad iff it forms no corner with the
# already-active set.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it))
    b = int(next(it))
    blocked = set()
    for _ in range(b):
        r = int(next(it)); c = int(next(it))
        blocked.add((r, c))

    P = set()
    # for incremental corner check keep column -> set of rows
    rows_in_col = {}

    def creates_corner(r, c):
        # adding (r,c). Corner patterns involving (r,c):
        # 1) (r,c) pivot: need (r+d,c) and (r,c+d) already in P.
        for (pr, pc) in P:
            if pc == c and pr > r:
                d = pr - r
                if (r, c + d) in P:
                    return True
            # 2) (r,c) = below-pad (r+d,c): pivot (r-d,c), need (r-d, c+d)? handled below
        # Rather than iterate all P (slow), do direct checks:
        return False

    # Direct O(m) per candidate check using membership set P.
    def ok(r, c):
        # (r,c) as pivot: exists d>=1 with (r+d,c) in P and (r,c+d) in P
        for d in range(1, m):
            if (r + d, c) in P and (r, c + d) in P:
                return False
        # (r,c) as vertical arm (r+d,c): pivot (r-d,c), other arm (r-d, c+d)
        for d in range(1, m):
            if (r - d, c) in P and (r - d, c + d) in P:
                return False
        # (r,c) as horizontal arm (r,c+d): pivot (r,c-d), other arm (r+d, c-d)
        for d in range(1, m):
            if (r, c - d) in P and (r + d, c - d) in P:
                return False
        return True

    for r in range(m):
        for c in range(m):
            if (r, c) in blocked:
                continue
            if ok(r, c):
                P.add((r, c))

    out = [str(len(P))]
    for (r, c) in sorted(P):
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
