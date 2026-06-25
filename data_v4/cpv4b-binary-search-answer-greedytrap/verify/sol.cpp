#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<long long> t;
int full;

vector<long long> sumMask;   // sumMask[mask] = total time of the jobs in mask

// Feasibility: can the jobs be partitioned into at most m presses,
// each press's total time <= cap?  This is the bin-packing DECISION problem.
// Correct DP: dp[mask] = minimum number of presses (each loaded <= cap)
// needed to cover exactly the jobs in `mask`. Transition: pick any sub-mask
// `sub` of the remaining jobs whose total <= cap to be ONE press, and recurse
// on the rest. We enumerate submasks of the complement.
bool feasible(long long cap) {
    for (int i = 0; i < n; i++) if (t[i] > cap) return false; // a job alone exceeds cap

    const int INF = 1e9;
    vector<int> dp(full + 1, INF);
    dp[0] = 0;
    for (int mask = 0; mask <= full; mask++) {
        if (dp[mask] == INF) continue;
        if (dp[mask] >= m) continue;          // already using m presses; can't open more
        int rest = full ^ mask;               // jobs not yet assigned
        // enumerate every non-empty submask of `rest` as the NEXT press's jobs
        for (int sub = rest; sub > 0; sub = (sub - 1) & rest) {
            if (sumMask[sub] <= cap) {
                int nmask = mask | sub;
                if (dp[mask] + 1 < dp[nmask]) dp[nmask] = dp[mask] + 1;
            }
        }
    }
    return dp[full] != INF && dp[full] <= m;
}

int main() {
    if (!(cin >> n >> m)) return 0;
    t.resize(n);
    for (auto &x : t) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }
    full = (1 << n) - 1;

    sumMask.assign(full + 1, 0);
    for (int mask = 1; mask <= full; mask++) {
        int low = mask & (-mask);                 // lowest set bit
        int idx = __builtin_ctz(low);
        sumMask[mask] = sumMask[mask ^ low] + t[idx];
    }

    long long lo = *max_element(t.begin(), t.end()); // no press below its biggest job
    long long hi = 0; for (long long x : t) hi += x;  // one press does everything
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) hi = mid; else lo = mid + 1;
    }
    cout << lo << "\n";
    return 0;
}
