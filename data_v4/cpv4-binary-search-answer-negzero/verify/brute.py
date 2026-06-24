import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    if n < k:
        print("INFEASIBLE")
        return

    # Independent brute force: enumerate EVERY contiguous block of length >= k,
    # compute its minimum element, and take the maximum of those minimums.
    best = None
    for i in range(n):
        for j in range(i, n):
            if (j - i + 1) >= k:
                m = min(a[i:j + 1])
                if best is None or m > best:
                    best = m
    print(best)

if __name__ == "__main__":
    main()
