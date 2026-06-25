#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    const long long MOD = 1000000007LL;

    int n;
    if (!(cin >> n)) return 0;            // no input -> nothing broadcast -> 0 reels
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    // dp = number of distinct subsequences (including the empty one) of the prefix seen so far.
    // Start with dp = 1: the empty sequence is the only distinct subsequence of the empty prefix.
    // For each new tone x: dp_new = 2*dp_old - (dp value right before the PREVIOUS occurrence of x).
    // The subtracted term removes subsequences that would otherwise be counted twice; if x is new,
    // nothing is subtracted. "last[x]" stores the dp value AS OF just before x was last appended,
    // i.e. the old dp at that earlier step (the count of distinct subsequences not yet using that x).
    long long dp = 1;                     // counts the empty subsequence
    unordered_map<long long, long long> last; // tone id -> dp snapshot to subtract on its next reuse
    last.reserve(n * 2);

    for (int i = 0; i < n; i++) {
        long long old = dp;
        dp = (2 * old) % MOD;
        auto it = last.find(t[i]);
        if (it != last.end()) {
            dp = (dp - it->second % MOD + MOD) % MOD;
        }
        // The next time tone t[i] appears, we must subtract exactly "old" (the dp value that held
        // immediately before appending THIS occurrence). Overwrite so only the latest occurrence counts.
        last[t[i]] = old;
    }

    // dp now includes the empty sequence; the problem wants non-empty reels, so subtract 1.
    long long ans = (dp - 1 + MOD) % MOD;
    cout << ans << "\n";
    return 0;
}
