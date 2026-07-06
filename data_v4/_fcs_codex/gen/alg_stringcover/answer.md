```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {
        return 0;
    }

    int n = static_cast<int>(s.size());
    const int INF = 1e9;

    vector<int> dp(n + 1, INF);
    dp[0] = 0;

    for (int center = 0; center < n; ++center) {
        for (int l = center, r = center; l >= 0 && r < n && s[l] == s[r]; --l, ++r) {
            dp[r + 1] = min(dp[r + 1], dp[l] + 1);
        }

        for (int l = center, r = center + 1; l >= 0 && r < n && s[l] == s[r]; --l, ++r) {
            dp[r + 1] = min(dp[r + 1], dp[l] + 1);
        }
    }

    cout << dp[n] << '\n';
    return 0;
}
```