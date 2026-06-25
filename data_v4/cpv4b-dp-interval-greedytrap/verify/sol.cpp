#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0

    // Each observation: half-open interval [s, e) with value v >= 0.
    struct Obs { long long s, e, v; };
    vector<Obs> obs(n);
    for (int i = 0; i < n; i++) cin >> obs[i].s >> obs[i].e >> obs[i].v;

    // Sort by end time ascending (ties broken by start, irrelevant for correctness).
    sort(obs.begin(), obs.end(), [](const Obs &a, const Obs &b) {
        if (a.e != b.e) return a.e < b.e;
        return a.s < b.s;
    });

    // ends[i] = finishing time of the i-th observation in sorted order.
    vector<long long> ends(n);
    for (int i = 0; i < n; i++) ends[i] = obs[i].e;

    // dp[i] = best total value using a subset of the first i sorted observations.
    // Half-open intervals: observation i (start obs[i].s) is compatible with any earlier
    // observation whose end <= obs[i].s, i.e. it does not overlap.
    vector<long long> dp(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long s = obs[i - 1].s;
        // p = number of observations (among the first i-1) whose end <= s.
        // Those are exactly indices [0, p) in sorted order; ends is sorted ascending.
        int p = (int)(upper_bound(ends.begin(), ends.begin() + (i - 1), s) - ends.begin());
        long long take = obs[i - 1].v + dp[p];   // include observation i-1
        long long skip = dp[i - 1];              // exclude it
        dp[i] = max(take, skip);
    }

    cout << dp[n] << "\n";
    return 0;
}
