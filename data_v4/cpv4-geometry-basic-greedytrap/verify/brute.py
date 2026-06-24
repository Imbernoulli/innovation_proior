import sys, functools

def solve(n, L, xs):
    # Minimum number of closed intervals of length L (each [s, s+L], s any integer)
    # needed to cover all integer points xs.
    #
    # INDEPENDENT and exhaustive: with integer coordinates and integer L, every
    # distinct coverage pattern of an interval is realized by an integer left
    # anchor s in the range [min(x)-1, max(x)+1]. We DP over sorted points: to
    # cover the leftmost uncovered point pts[i], we try EVERY integer anchor s in
    # that full range whose interval [s, s+L] covers pts[i], and recurse. This
    # makes NO greedy assumption -- it brute-forces the whole discrete placement
    # space and keeps the minimum.
    if n == 0:
        return 0
    pts = sorted(xs)
    lo = min(pts) - 1
    hi = max(pts) + 1

    sys.setrecursionlimit(100000)
    INF = float('inf')

    @functools.lru_cache(maxsize=None)
    def dp(i):
        if i >= n:
            return 0
        best = INF
        # interval [s, s+L] must cover pts[i]: s <= pts[i] <= s+L
        slo = max(lo, pts[i] - L)
        shi = min(hi, pts[i])
        s = slo
        while s <= shi:
            right = s + L
            j = i
            while j < n and pts[j] <= right:
                j += 1
            r = dp(j)
            if r + 1 < best:
                best = r + 1
            s += 1
        return best

    return dp(0)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    xs = [int(data[idx + t]) for t in range(n)]
    print(solve(n, L, xs))

if __name__ == "__main__":
    main()
