#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0: already across, cost 0
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // dp[i] = minimum total stamina to be standing on stone i, having legally
    // arrived there from the near bank. You may first land on stone 0 or stone 1.
    // From stone j you may have come from stone j-1 or stone j-2.
    // Far bank is reachable from stone n-1 or stone n-2 at no extra cost.
    if (n == 0) { cout << 0 << "\n"; return 0; }

    vector<long long> dp(n);
    dp[0] = c[0];                          // first leap lands on stone 0
    if (n >= 2) dp[1] = min(c[1], dp[0] + c[1]); // first leap onto 1, or step 0->1
    for (int i = 2; i < n; i++) {
        dp[i] = min(dp[i - 1], dp[i - 2]) + c[i];
    }

    // reach far bank from stone n-1 or stone n-2
    long long ans = dp[n - 1];
    if (n >= 2) ans = min(ans, dp[n - 2]);

    cout << ans << "\n";
    return 0;
}
