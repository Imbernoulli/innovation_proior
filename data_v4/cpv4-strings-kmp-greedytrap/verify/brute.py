#!/usr/bin/env python3
# Independent, obviously-correct brute force for the SAME problem:
#   Split t into consecutive blocks, each block equal to some non-empty prefix of s.
#   Minimize the number of blocks. Print -1 if impossible.
import sys

def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        return
    s = data[0]
    t = data[1]
    n = len(t)
    m = len(s)
    INF = float('inf')
    # dp[i] = min blocks to exactly cover t[0:i].
    dp = [INF] * (n + 1)
    dp[0] = 0
    for i in range(n):
        if dp[i] == INF:
            continue
        # try EVERY prefix length L of s as the next block, no shortcuts
        for L in range(1, m + 1):
            if i + L <= n and t[i:i+L] == s[:L]:
                if dp[i] + 1 < dp[i + L]:
                    dp[i + L] = dp[i] + 1
    print(dp[n] if dp[n] != INF else -1)

if __name__ == "__main__":
    main()
