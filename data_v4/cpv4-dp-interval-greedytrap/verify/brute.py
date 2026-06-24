import sys
from functools import lru_cache

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    w = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Brute force: enumerate EVERY way to partition the n beads into contiguous
    # nonempty lines, reject any partition that has a line exceeding width W,
    # compute total penalty (squared slack per line, last line free), take the min.
    #
    # A partition is a choice of cut positions among the n-1 gaps between beads.
    # We enumerate all 2^(n-1) subsets of cut positions.

    INF = float('inf')

    def used_width(lo, hi):  # beads lo..hi inclusive (0-based)
        cnt = hi - lo + 1
        return sum(w[lo:hi + 1]) + (cnt - 1)

    best = INF
    gaps = n - 1  # positions between consecutive beads
    for mask in range(1 << gaps):
        # cut after bead p (0-based) if bit p set; build segments
        segs = []
        start = 0
        ok = True
        for p in range(gaps):
            if mask & (1 << p):
                segs.append((start, p))
                start = p + 1
        segs.append((start, n - 1))
        total = 0
        for si, (lo, hi) in enumerate(segs):
            u = used_width(lo, hi)
            if u > W:
                ok = False
                break
            is_last = (si == len(segs) - 1)
            if not is_last:
                slack = W - u
                total += slack * slack
        if ok:
            best = min(best, total)

    print(best if best != INF else -1)

if __name__ == "__main__":
    main()
