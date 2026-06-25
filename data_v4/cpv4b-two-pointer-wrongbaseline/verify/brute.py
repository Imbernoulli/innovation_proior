import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    S = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Independent O(n^2) brute force: try every contiguous window [l, r],
    # accumulate its sum directly, and track the shortest length with sum >= S.
    # We do NOT break early: with negative values a longer window from the same
    # l can dip below S and rise again, so every (l, r) must be examined.
    best = None
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            if s >= S:
                length = r - l + 1
                if best is None or length < best:
                    best = length
    print(-1 if best is None else best)

if __name__ == "__main__":
    main()
