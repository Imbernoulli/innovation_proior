#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long B;
    long long J;
    if (!(cin >> n >> B >> J)) return 0;

    const long long MOD = 1000000007LL;

    vector<long long> price(n), joy(n);
    for (int i = 0; i < n; i++) cin >> price[i] >> joy[i];

    // dp[c][j] = number of distinct subsets (0/1) chosen so far whose total
    // price == c and whose joy CLAMPED at J equals j.
    // Joy is clamped to J because we only care about "joy >= J": every subset
    // with true joy >= J collapses into the bucket j = J, so the answer is the
    // single cell dp[B][J] at the end. Clamping keeps the second dimension O(J).
    int Bc = (int)B;
    int Jc = (int)J;

    // dp indexed [price 0..Bc][clampedJoy 0..Jc]
    vector<vector<long long>> dp(Bc + 1, vector<long long>(Jc + 1, 0));
    dp[0][0] = 1; // the empty subset: price 0, joy 0

    for (int i = 0; i < n; i++) {
        long long p = price[i];
        long long jv = joy[i];
        if (p > Bc) continue; // cannot ever fit by price
        // 0/1 knapsack: iterate price DOWNWARD so each item is used at most once.
        for (int c = Bc; c >= (int)p; c--) {
            // iterate clamped joy DOWNWARD as well, same 0/1 reason.
            for (int j = Jc; j >= 0; j--) {
                if (dp[c - (int)p][j] == 0) continue;
                int nj = j + (int)jv;
                if (nj > Jc) nj = Jc; // clamp: joy >= J all collapse to bucket J
                dp[c][nj] = (dp[c][nj] + dp[c - (int)p][j]) % MOD;
            }
        }
    }

    cout << dp[Bc][Jc] % MOD << "\n";
    return 0;
}
