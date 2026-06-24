import sys

MOD = 1000000007

def main():
    data = sys.stdin.read().split()
    n, k, c = int(data[0]), int(data[1]), int(data[2])

    # A multiset of size k from n distinguishable colors with each color used at most c times
    # is exactly an n-tuple (x_1,...,x_n) with 0 <= x_i <= c and sum x_i = k.
    # Count such tuples by a bounded stars-and-bars DP over colors.
    dp = [0] * (k + 1)
    dp[0] = 1
    for _color in range(n):
        ndp = [0] * (k + 1)
        for s in range(k + 1):
            cur = dp[s]
            if cur == 0:
                continue
            use = 0
            while use <= c and s + use <= k:
                ndp[s + use] = (ndp[s + use] + cur) % MOD
                use += 1
        dp = ndp
    print(dp[k] % MOD)

main()
