#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // GOAL: output a subset of the n items whose weights sum to EXACTLY S using the
    // FEWEST possible items. If several minimum-size subsets exist, any one is accepted.
    // If no subset of the items sums to S, output the single line: -1
    //
    // Output on success:
    //   line 1: k  (number of chosen items, k >= 0)
    //   line 2: k distinct 1-based indices in ascending order (empty line if k == 0)
    //
    // S may be 0, in which case the empty subset (k = 0) is the unique minimum.

    const int SZ = (int)S;
    const int INF = 1e9;

    // dp[s] = minimum #items (using items considered so far) to reach sum s exactly.
    // take[i][s] = true iff, when item i was processed, it strictly improved dp[s]
    //              (i.e. the optimal way to reach s using items 0..i ENDS by adding item i).
    // The back-table lets us reconstruct ONE consistent minimum-count subset without
    // ever reusing an item: from (i=n-1, s=S) we either step to (i-1, s) when take[i][s]
    // is false, or take item i and step to (i-1, s-w[i]) when it is true.
    vector<int> dp(SZ + 1, INF);
    dp[0] = 0;
    // take stored as a flat vector<char> of size n*(S+1).
    vector<char> take((size_t)n * (SZ + 1), 0);

    for (int i = 0; i < n; i++) {
        char *row = &take[(size_t)i * (SZ + 1)];
        if (w[i] <= 0 || w[i] > S) continue; // cannot help reach a positive sum <= S
        int wi = (int)w[i];
        for (int s = SZ; s >= wi; s--) {
            if (dp[s - wi] != INF && dp[s - wi] + 1 < dp[s]) {
                dp[s] = dp[s - wi] + 1;
                row[s] = 1;
            }
        }
    }

    if (dp[SZ] == INF) { cout << -1 << "\n"; return 0; }

    // Reconstruct by walking the back-table from (n-1, S) down to (-1, 0).
    vector<int> chosen;
    int s = SZ;
    for (int i = n - 1; i >= 0 && s > 0; i--) {
        const char *row = &take[(size_t)i * (SZ + 1)];
        if (row[s]) {
            chosen.push_back(i + 1); // 1-based
            s -= (int)w[i];
        }
    }
    // s is now 0 and every chosen index is distinct (each item visited at most once).
    sort(chosen.begin(), chosen.end());

    cout << (int)chosen.size() << "\n";
    for (size_t k = 0; k < chosen.size(); k++) {
        cout << chosen[k];
        cout << (k + 1 == chosen.size() ? '\n' : ' ');
    }
    if (chosen.empty()) cout << "\n"; // empty second line for k == 0
    return 0;
}
