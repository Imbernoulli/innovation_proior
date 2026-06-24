import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    # DP: dp[k] = minimum number of squares summing to k.
    # Obviously correct unbounded-style DP over squares <= k.
    INF = float('inf')
    dp = [0] + [INF] * n
    squares = []
    i = 1
    while i * i <= n:
        squares.append(i * i)
        i += 1
    for k in range(1, n + 1):
        best = INF
        for s in squares:
            if s > k:
                break
            v = dp[k - s] + 1
            if v < best:
                best = v
        dp[k] = best
    print(dp[n])

main()
