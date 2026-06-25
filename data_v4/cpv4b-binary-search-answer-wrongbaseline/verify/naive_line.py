import sys

def feasible_line(p, n, k, L, d):
    # NAIVE: treat as a line. Anchor the first chosen post at p[0], sweep,
    # place greedily whenever the gap from the last placed >= d. Count how
    # many we can place. Feasible iff we can place >= k.
    # (This is the standard 'aggressive cows' line greedy.)
    count = 1
    last = p[0]
    for i in range(1, n):
        if p[i] - last >= d:
            count += 1
            last = p[i]
            if count >= k:
                return True
    return count >= k

def solve(data):
    it = iter(data)
    n = next(it); k = next(it); L = next(it)
    p = [next(it) for _ in range(n)]
    lo, hi = 0, L  # clearance in [0, L//2] really, but search up to L
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible_line(p, n, k, L, mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans

def main():
    data = list(map(int, sys.stdin.read().split()))
    print(solve(data))

if __name__ == "__main__":
    main()
