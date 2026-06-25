#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long W;
    int M, q;
    if (!(cin >> n >> W >> M >> q)) return 0;

    vector<long long> w(n);
    vector<int> p(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> p[i];

    // Masses beyond W are useless; cap the table at W.
    // dp[mass][phase] = number of subsets with total mass == mass and total
    // phase (mod M) == phase. Two dimensions are coupled, so we carry both.
    int Wcap = (int)W; // W fits in int by constraints (<= 2000)
    vector<vector<long long>> dp(Wcap + 1, vector<long long>(M, 0));
    dp[0][0] = 1; // empty subset

    for (int i = 0; i < n; i++) {
        long long wi = w[i];
        int pi = p[i] % M;
        if (wi > Wcap) continue; // cannot fit, skip (still counts as "not taken")
        int wic = (int)wi;
        // 0/1 knapsack: iterate mass downward to avoid reusing item i.
        for (int m = Wcap; m >= wic; m--) {
            const vector<long long> &src = dp[m - wic];
            vector<long long> &dst = dp[m];
            for (int ph = 0; ph < M; ph++) {
                long long v = src[ph];
                if (v == 0) continue;
                int nph = ph + pi;
                if (nph >= M) nph -= M;
                dst[nph] += v;
                if (dst[nph] >= MOD) dst[nph] -= MOD;
            }
        }
    }

    cout << dp[Wcap][q] % MOD << "\n";
    return 0;
}
