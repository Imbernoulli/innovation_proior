import sys

def solve(data):
    it = iter(data)
    n = int(next(it))
    L = int(next(it))
    a = [int(next(it)) for _ in range(n)]

    # f[i] = best total over days a[i..n-1]. Choose, at each position i, either:
    #   - leave day i uncovered: f[i+1]
    #   - start an interval [i, k-1] of length (k - i) >= L: sum(a[i..k-1]) + f[k]
    # Answer is f[0]. This is an exhaustive recursion over all valid placements,
    # independent of the prefix-sum + incremental-max method in sol.cpp.
    NEG = float('-inf')
    f = [0] * (n + 1)  # f[n] = 0 (no days left, take nothing)
    for i in range(n - 1, -1, -1):
        best = f[i + 1]  # leave day i uncovered
        s = 0
        for k in range(i + 1, n + 1):
            s += a[k - 1]            # s = sum(a[i..k-1]), length = k - i
            if k - i >= L:
                cand = s + f[k]
                if cand > best:
                    best = cand
        f[i] = best
    return f[0]

def main():
    data = sys.stdin.read().split()
    print(solve(data))

if __name__ == "__main__":
    main()
