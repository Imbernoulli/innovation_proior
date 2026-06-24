#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    if (!(cin >> m >> n)) return 0;          // m = number of slots, n = number of bands

    // For each band, build the bitmask of slots it occupies. A booked band starting at
    // slot s with duration d covers the HALF-OPEN interval [s, s+d): slots
    // s, s+1, ..., s+d-1. It is bookable only if it fits entirely inside [0, m), which
    // means s + d <= m (equivalently its last slot s+d-1 <= m-1). Bands that run past
    // the last slot are simply unbookable and are dropped.
    // Group usable bands by their lowest occupied slot so the DP only scans relevant ones.
    vector<vector<pair<int,long long>>> byLow(m); // byLow[s] = {(mask, profit), ...}
    for (int i = 0; i < n; i++) {
        long long s, d, p;
        cin >> s >> d >> p;
        if (d <= 0 || s < 0 || s + d > m) continue;   // does not fit in [0, m)
        int mask = 0;
        for (long long k = s; k < s + d; k++) mask |= (1 << (int)k);   // [s, s+d)
        long long prof = (p > 0 ? p : 0); // a non-positive band is never worth booking
        byLow[(int)s].push_back({mask, prof});
    }

    int full = 1 << m;
    // dp[mask] = max total profit using pairwise slot-disjoint bands whose union of
    // occupied slots is a SUBSET of mask. dp[0] = 0 (book nothing). We fill masks in
    // increasing order; from each mask we look at its lowest FREE slot `low` and either
    // leave it empty, or start one band there (a band whose lowest slot is exactly low).
    vector<long long> dp(full, 0);
    for (int mask = 0; mask < full; mask++) {
        long long cur = dp[mask];
        int low = -1;
        for (int b = 0; b < m; b++) if (!(mask & (1 << b))) { low = b; break; }
        if (low == -1) continue;                       // every slot occupied
        int leaveEmpty = mask | (1 << low);            // option: slot `low` stays empty
        if (cur > dp[leaveEmpty]) dp[leaveEmpty] = cur;
        for (auto &pr : byLow[low]) {                  // bands whose first slot is `low`
            int bm = pr.first;
            if (bm & mask) continue;                   // overlaps an occupied slot
            int nm = mask | bm;
            long long val = cur + pr.second;
            if (val > dp[nm]) dp[nm] = val;
        }
    }

    long long ans = 0;
    for (int mask = 0; mask < full; mask++) ans = max(ans, dp[mask]);
    cout << ans << "\n";
    return 0;
}
