// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Genuine insight: (1) use the EXACT cubic inverse (not a linear tangent
// approximation) to know each column's per-cost error curve exactly, then
// (2) recognize the shared slit budget turns this into a multiple-choice
// knapsack: for every column pick one cost u in [0, k-S] (one "menu option"),
// minimizing total error subject to sum(u) <= C. Solved by an exact DP over
// the shared budget, which correctly balances columns instead of greedily
// exhausting the budget on whichever column looks neediest first.
int main() {
    int m, k, S;
    ll M, C;
    cin >> m >> k >> S >> M >> C;
    vector<double> t(m);
    for (int i = 0; i < m; i++) { ll x; cin >> x; t[i] = x / 1000.0; }

    int maxU = k - S;
    int Ci = (int)C;

    const double INF = 1e18;
    vector<double> dp(Ci + 1, INF);
    dp[0] = 0.0;
    vector<vector<int>> choice(m, vector<int>(Ci + 1, 0));

    vector<double> err(maxU + 1);
    for (int col = 0; col < m; col++) {
        for (int u = 0; u <= maxU; u++) {
            int r = k - u;
            double curv = (double)M / ((double)r * r * r);
            err[u] = fabs(curv - t[col]);
        }
        vector<double> ndp(Ci + 1, INF);
        for (int j = 0; j <= Ci; j++) {
            double best = INF;
            int bestu = 0;
            int umax = min(maxU, j);
            for (int u = 0; u <= umax; u++) {
                if (dp[j - u] >= INF / 2) continue;
                double cand = dp[j - u] + err[u];
                if (cand < best) { best = cand; bestu = u; }
            }
            ndp[j] = best;
            choice[col][j] = bestu;
        }
        dp.swap(ndp);
    }

    int bestj = 0;
    double bestval = INF;
    for (int j = 0; j <= Ci; j++) if (dp[j] < bestval) { bestval = dp[j]; bestj = j; }

    vector<int> u(m);
    int j = bestj;
    for (int col = m - 1; col >= 0; col--) {
        int uu = choice[col][j];
        u[col] = uu;
        j -= uu;
    }

    for (int i = 0; i < m; i++) cout << (k - u[i]) << (i + 1 == m ? '\n' : ' ');
    return 0;
}
