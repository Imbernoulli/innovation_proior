#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> cal(n), mass(n);
    long long sumMass = 0;
    for (int i = 0; i < n; i++) { cin >> cal[i] >> mass[i]; sumMass += mass[i]; }

    const long long NEG = LLONG_MIN / 4;
    // dp[w] = max total cal over a subset whose total mass is EXACTLY w (NEG = unreachable).
    vector<long long> dp(sumMass + 1, NEG);
    dp[0] = 0; // empty subset: mass 0, value 0
    for (int i = 0; i < n; i++) {
        for (long long w = sumMass; w >= mass[i]; w--) {
            if (dp[w - mass[i]] > NEG) {
                long long cand = dp[w - mass[i]] + cal[i];
                if (cand > dp[w]) dp[w] = cand;
            }
        }
    }

    // Among masses W in [L, sumMass] reachable by a NON-EMPTY subset, maximize value/W.
    // Best fraction bestP/bestW maximized; bestP = -1 marks "none yet".
    long long bestP = -1, bestW = 1;
    for (long long W = L; W <= sumMass; W++) {
        if (dp[W] <= NEG) continue;       // mass W not reachable
        if (W == 0) continue;             // density undefined; L>=1 anyway
        long long P = dp[W];
        // compare P/W vs bestP/bestW using exact 128-bit cross multiplication
        if (bestP < 0) { bestP = P; bestW = W; continue; }
        __int128 lhs = (__int128)P * bestW;
        __int128 rhs = (__int128)bestP * W;
        if (lhs > rhs) { bestP = P; bestW = W; }
    }

    if (bestP < 0) { cout << "IMPOSSIBLE\n"; return 0; }
    long long g = std::__gcd(bestP, bestW);
    if (g == 0) g = 1;
    cout << (bestP / g) << "/" << (bestW / g) << "\n";
    return 0;
}
