#include <bits/stdc++.h>
using namespace std;
int main() {
    int n; long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> cal(n), mass(n);
    long long sumMass = 0;
    for (int i = 0; i < n; i++) { cin >> cal[i] >> mass[i]; sumMass += mass[i]; }
    const long long NEG = LLONG_MIN / 4;
    vector<long long> dp(sumMass + 1, NEG);
    dp[0] = 0;
    for (int i = 0; i < n; i++)
        for (long long w = sumMass; w >= mass[i]; w--)
            if (dp[w - mass[i]] > NEG) {
                long long cand = dp[w - mass[i]] + cal[i];
                if (cand > dp[w]) dp[w] = cand;
            }
    long long bestP = -1, bestW = 1;
    for (long long W = L; W <= sumMass; W++) {
        if (dp[W] <= NEG) continue;
        long long P = dp[W];
        if (bestP < 0) { bestP = P; bestW = W; continue; }
        long long lhs = P * bestW;        // BUG: 64-bit overflow
        long long rhs = bestP * W;        // BUG: 64-bit overflow
        if (lhs > rhs) { bestP = P; bestW = W; }
    }
    if (bestP < 0) { cout << "IMPOSSIBLE\n"; return 0; }
    long long g = std::__gcd(bestP, bestW);
    if (g == 0) g = 1;
    cout << (bestP / g) << "/" << (bestW / g) << "\n";
    return 0;
}
