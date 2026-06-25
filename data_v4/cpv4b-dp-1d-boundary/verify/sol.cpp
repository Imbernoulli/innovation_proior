#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    const long long INF = LLONG_MAX / 4;

    // dp[j] = minimum total toll to be standing on stone j, having started on
    // stone 0 (whose toll is paid). A leap from i to j is legal iff stone j is
    // not broken and 1 <= j - i <= D (the reach D is INCLUSIVE: landing exactly
    // D ahead is allowed). A toll c[j] < 0 marks a broken stone (cannot land).
    vector<long long> dp(n, INF);

    // Stone 0 is the start. If it is broken the frog cannot even begin.
    if (c[0] >= 0) dp[0] = c[0];

    // Sliding-window minimum over the legal predecessor range [j-D, j-1].
    // The deque holds indices i with increasing index and increasing dp[i].
    deque<int> dq;
    for (int j = 0; j < n; j++) {
        // Drop predecessors that are now out of reach: i < j - D means the gap
        // j - i > D, which is illegal. The boundary i == j - D stays (gap == D).
        while (!dq.empty() && dq.front() < j - D) dq.pop_front();

        if (j > 0 && c[j] >= 0 && !dq.empty() && dp[dq.front()] < INF)
            dp[j] = dp[dq.front()] + c[j];

        // Push j as a future predecessor only if we can actually stand on it.
        if (dp[j] < INF) {
            while (!dq.empty() && dp[dq.back()] >= dp[j]) dq.pop_back();
            dq.push_back(j);
        }
    }

    if (dp[n - 1] >= INF) cout << -1 << "\n";
    else cout << dp[n - 1] << "\n";
    return 0;
}
