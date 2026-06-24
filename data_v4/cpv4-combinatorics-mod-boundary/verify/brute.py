import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    k, c, S, M = (int(x) for x in data[:4])

    # Brute force: count integer solutions to x_1 + ... + x_k = S with 0 <= x_i <= c,
    # by a straightforward bounded-composition DP, then reduce modulo M at the end.
    # dp[s] = number of ways using the children processed so far to reach partial sum s.
    if S < 0:
        print(0)
        return
    if k == 0:
        print((1 % M) if S == 0 else 0)
        return
    if S > k * c:
        print(0)
        return

    dp = [0] * (S + 1)
    dp[0] = 1
    for _ in range(k):
        # add one bounded variable taking values 0..c, using prefix sums (exact ints)
        pref = [0] * (S + 2)
        for s in range(S + 1):
            pref[s + 1] = pref[s] + dp[s]
        ndp = [0] * (S + 1)
        for s in range(S + 1):
            lo = s - c
            if lo < 0:
                lo = 0
            ndp[s] = pref[s + 1] - pref[lo]
        dp = ndp
    print(dp[S] % M)

if __name__ == "__main__":
    main()
