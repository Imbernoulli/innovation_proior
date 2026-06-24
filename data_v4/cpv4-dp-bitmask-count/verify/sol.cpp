#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;

    int full = (n > 0) ? ((1 << n) - 1) : 0;

    // allowed[mask] = 1 iff `mask` is one of the candidate squads, AFTER
    // discarding empty squads, masking off out-of-range bits, and de-duplicating
    // (a repeated mask is still just one allowed squad).
    vector<char> allowed(full + 1, 0);
    for (int j = 0; j < m; j++) {
        int x;
        if (scanf("%d", &x) != 1) x = 0;
        x &= full;             // drop any bit >= n
        if (x == 0) continue;  // empty squad is not a real squad
        allowed[x] = 1;        // marking twice is harmless -> dedup for free
    }

    // dp[mask] = number of ways to partition the employee set `mask` into a
    // collection of candidate squads (each used at most once, order irrelevant),
    // modulo MOD. dp[0] = 1: the empty set has exactly one partition (use no
    // squads).
    //
    // Canonical order to avoid double-counting: the squad that owns the LOWEST
    // remaining employee is decided first. So we only enumerate squads `sub`
    // that are submasks of `mask` AND contain mask's lowest set bit. Each
    // unordered partition is then generated exactly once.
    vector<long long> dp(full + 1, 0);
    dp[0] = 1;
    for (int mask = 1; mask <= full; mask++) {
        int low = mask & (-mask);          // lowest set bit of mask
        long long ways = 0;
        for (int sub = mask; sub; sub = (sub - 1) & mask) {
            if (!(sub & low)) continue;    // squad must own the lowest employee
            if (allowed[sub]) ways += dp[mask ^ sub];
        }
        dp[mask] = ways % MOD;
    }

    printf("%lld\n", dp[full] % MOD);
    return 0;
}
