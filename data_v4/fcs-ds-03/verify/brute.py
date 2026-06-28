#!/usr/bin/env python3
# Independent O(n^2) DP oracle for fcs-ds-03.
#   dp[0] = 0
#   dp[i] = c[i] + min_{0 <= j < i} ( dp[j] + b[j]*a[i] )   for i = 1..n
#   answer = dp[n]
# Reads:  n ; a[1..n] ; b[0..n-1] ; c[1..n]   (whitespace separated)
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    if n == 0:
        print(0)
        return
    a = [0] + [int(next(it)) for _ in range(n)]   # a[1..n]
    b = [int(next(it)) for _ in range(n)]         # b[0..n-1]
    c = [0] + [int(next(it)) for _ in range(n)]   # c[1..n]

    INF = float('inf')
    dp = [INF] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        best = INF
        for j in range(0, i):
            best = min(best, dp[j] + b[j] * a[i])
        dp[i] = c[i] + best
    print(dp[n])

main()
