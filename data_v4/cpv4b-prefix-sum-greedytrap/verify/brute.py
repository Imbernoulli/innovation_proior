import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]

    # Independent O(n^2) DP brute force over ALL partitions of the whole array
    # into contiguous non-empty blocks; maximize the count of blocks with sum > 0.
    # prefix[i] = sum of first i elements.
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + a[i]

    NEG = -10 ** 18
    dp = [NEG] * (n + 1)
    dp[0] = 0  # empty prefix, zero blocks
    for i in range(1, n + 1):
        for j in range(0, i):  # last block is (j, i]
            if dp[j] == NEG:
                continue
            seg = prefix[i] - prefix[j]
            val = dp[j] + (1 if seg > 0 else 0)
            if val > dp[i]:
                dp[i] = val
    print(dp[n] if n > 0 else 0)

main()
