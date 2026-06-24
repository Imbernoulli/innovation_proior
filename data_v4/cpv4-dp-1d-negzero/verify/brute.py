import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    c = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Exhaustively try every non-empty contiguous block [l, r], profit = sum(a[l..r]) - c.
    # Also allow doing nothing (profit 0). Take the maximum.
    best = 0  # do nothing
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            best = max(best, s - c)
    print(best)

if __name__ == "__main__":
    main()
