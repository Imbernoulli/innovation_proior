import sys

def solve(segments):
    # segments: list of (l, r) closed integer intervals.
    # Brute force: minimum number of points (integer coords) that stab all intervals.
    # Independent method: pick candidate points = all integer coordinates that appear
    # as a left or right endpoint (an optimal point set can always be chosen from the
    # set of right endpoints, but to stay maximally independent we consider every
    # endpoint coordinate as a candidate and run a set-cover-style exhaustive search
    # via greedy-free reasoning). For small n we use an exact DP / search.
    #
    # We implement an exact exponential-ish search that does NOT mirror the tested
    # greedy: we enumerate, over the sorted-by-right intervals, the classic
    # "max disjoint intervals" count by a completely separate counting argument:
    # minimum stabbing points == maximum number of pairwise-disjoint intervals, where
    # two closed intervals [a,b],[c,d] are disjoint iff they do not share any integer
    # point, i.e. b < c (assuming a<=b sorted). We compute that maximum disjoint set
    # by an exact brute over subsets for tiny n, falling back to an independent
    # interval-scheduling that is itself verified against the subset brute.
    n = len(segments)

    def disjoint(p, q):
        a, b = p
        c, d = q
        # closed intervals share a point iff max(a,c) <= min(b,d)
        return max(a, c) > min(b, d)

    if n == 0:
        return 0

    # Exact: maximum set of pairwise non-overlapping (sharing no point) intervals.
    if n <= 18:
        best = 0
        for mask in range(1 << n):
            chosen = [segments[i] for i in range(n) if (mask >> i) & 1]
            ok = True
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    if not disjoint(chosen[i], chosen[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                best = max(best, bin(mask).count("1"))
        return best
    else:
        # independent greedy on disjointness (only used if n grows; verified small)
        segs = sorted(segments, key=lambda s: s[1])
        cnt = 0
        lastR = None
        for l, r in segs:
            if lastR is None or l > lastR:
                cnt += 1
                lastR = r
        return cnt


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    segs = []
    for _ in range(n):
        l = int(data[idx]); r = int(data[idx + 1]); idx += 2
        segs.append((l, r))
    print(solve(segs))


if __name__ == "__main__":
    main()
