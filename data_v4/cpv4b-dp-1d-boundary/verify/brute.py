import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    D = int(data[idx]); idx += 1
    c = []
    for _ in range(n):
        c.append(int(data[idx])); idx += 1

    INF = float('inf')
    # dp[j] = min toll to stand on stone j starting from stone 0.
    # Broken stone: c[j] < 0 (cannot land there).
    # Leap i -> j legal iff 1 <= j - i <= D and j not broken.
    dp = [INF] * n
    if c[0] >= 0:
        dp[0] = c[0]
    for j in range(1, n):
        if c[j] < 0:
            continue
        best = INF
        # predecessors i with 1 <= j - i <= D  ==>  i in [j-D, j-1]
        lo = max(0, j - D)
        for i in range(lo, j):
            if dp[i] < INF:
                if dp[i] < best:
                    best = dp[i]
        if best < INF:
            dp[j] = best + c[j]

    print(-1 if dp[n - 1] == INF else dp[n - 1])

main()
