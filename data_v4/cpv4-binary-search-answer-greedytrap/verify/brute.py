import sys

def solve(n, k, a):
    # Minimum possible maximum block-sum when partitioning a[0..n-1] into at most
    # k contiguous blocks. Independent exhaustive DP (no binary search, no greedy).
    #
    # dp[j][i] = minimum achievable maximum block-sum for the prefix a[0..i-1]
    #            using exactly j blocks. We allow up to k blocks; using fewer than
    #            the available blocks is never worse, so "at most k" == answer at j=k
    #            (extra blocks could be empty, but empty blocks have sum 0 which never
    #            increases the max; to keep the DP simple we forbid empty blocks and
    #            instead take the min over j = 1..min(k, n)).
    if n == 0:
        return 0
    INF = float('inf')
    # prefix sums
    pre = [0] * (n + 1)
    for i in range(n):
        pre[i + 1] = pre[i] + a[i]

    kk = min(k, n)  # never need more than n blocks
    # dp[i] using current number of blocks; iterate blocks from 1..kk
    # dp1[i] = max block sum for prefix length i with exactly j blocks
    # base: j = 1
    dp = [INF] * (n + 1)
    dp[0] = 0  # empty prefix, 0 blocks placed so far baseline
    # exactly 1 block covering prefix i = whole-sum
    cur = [INF] * (n + 1)
    for i in range(1, n + 1):
        cur[i] = pre[i] - pre[0]
    best_for_blocks = [None] * (kk + 1)
    best_for_blocks[1] = cur[n]
    prev = cur
    for j in range(2, kk + 1):
        cur = [INF] * (n + 1)
        for i in range(1, n + 1):
            # last block is a[t..i-1], previous j-1 blocks cover a[0..t-1]
            best = INF
            for t in range(j - 1, i):  # at least j-1 beds for j-1 blocks
                if prev[t] == INF:
                    continue
                seg = pre[i] - pre[t]
                cand = prev[t] if prev[t] > seg else seg
                if cand < best:
                    best = cand
            cur[i] = best
        best_for_blocks[j] = cur[n]
        prev = cur

    ans = INF
    for j in range(1, kk + 1):
        if best_for_blocks[j] is not None and best_for_blocks[j] < ans:
            ans = best_for_blocks[j]
    return ans

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    print(solve(n, k, a))

if __name__ == "__main__":
    main()
