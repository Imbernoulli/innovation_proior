import sys

MOD = 10**9 + 7

def popcount(x):
    return bin(x).count("1")

def solve(data):
    n = data[0]
    a = data[1:1+n]
    # dp[i] = number of clean partitions of prefix a[0..i-1]
    # A partition is clean iff every contiguous segment has even-popcount XOR signature.
    # Brute force directly from the definition: for each split position j < i, the last
    # segment is a[j..i-1] with XOR signature s; it must have even popcount.
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        s = 0
        total = 0
        # last segment runs from j..i-1 (1-indexed boundary j)
        for j in range(i - 1, -1, -1):
            s ^= a[j]
            if popcount(s) % 2 == 0:
                total += dp[j]
        dp[i] = total % MOD
    return dp[n] % MOD

def main():
    data = list(map(int, sys.stdin.read().split()))
    if not data:
        return
    print(solve(data))

if __name__ == "__main__":
    main()
