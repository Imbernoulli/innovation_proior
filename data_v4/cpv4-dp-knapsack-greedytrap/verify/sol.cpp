#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;

    vector<long long> c(n), r(n);
    for (int i = 0; i < n; i++) cin >> c[i] >> r[i];

    // dp[w] = best total yield achievable with total cost <= w.
    // Process each assay once; iterate w from high to low so dp[w - c[i]]
    // still reflects "this assay not yet used" -> each assay counts at most once.
    vector<long long> dp(B + 1, 0);
    for (int i = 0; i < n; i++) {
        long long ci = c[i], ri = r[i];
        if (ci > B) continue;                 // never fits, skip
        for (long long w = B; w >= ci; w--) {
            long long cand = dp[w - ci] + ri; // run assay i, leaving budget w - ci
            if (cand > dp[w]) dp[w] = cand;
        }
    }

    cout << dp[B] << "\n";                     // "<= B" semantics fold in unspent budget
    return 0;
}
