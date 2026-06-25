#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // empty input -> no batches -> cost 0

    // c[i][j] = cleaning cost when batch j runs immediately after batch i (i is the previous run).
    // e[i][k] = extra carry-over penalty on batch k when batch i ran two positions earlier.
    vector<vector<long long>> c(n, vector<long long>(n, 0));
    vector<vector<long long>> e(n, vector<long long>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cin >> c[i][j];
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) cin >> e[i][k];

    if (n == 0) { cout << 0 << "\n"; return 0; }
    if (n == 1) { cout << 0 << "\n"; return 0; }   // a single run: no transitions at all

    const long long INF = (long long)4e18;
    int full = 1 << n;

    // dp[mask][last][prev]: minimum cost of a sequence whose run-set is exactly `mask`,
    // whose most recent batch is `last`, and whose batch before that is `prev`.
    // We must remember the previous TWO batches because the carry-over penalty e[][]
    // depends on the batch two positions back, which a single-`last` state cannot supply.
    // Layout dp[mask][last*n + prev]. prev == last is used to mark "only one batch placed
    // so far" (no batch two positions back exists yet).
    static vector<vector<long long>> dp;
    dp.assign(full, vector<long long>(n * n, INF));

    // Length-1 prefixes: place a single batch s first. Mark prev == last (no two-back yet).
    for (int s = 0; s < n; s++) {
        dp[1 << s][s * n + s] = 0;
    }

    long long answer = INF;

    for (int mask = 1; mask < full; mask++) {
        for (int last = 0; last < n; last++) {
            if (!(mask & (1 << last))) continue;
            for (int prev = 0; prev < n; prev++) {
                long long cur = dp[mask][last * n + prev];
                if (cur >= INF) continue;
                int pc = __builtin_popcount((unsigned)mask);
                if (pc == n) {                // completed sequence over all batches
                    answer = min(answer, cur);
                    continue;
                }
                bool single = (prev == last); // only one batch placed so far -> no two-back yet
                for (int nxt = 0; nxt < n; nxt++) {
                    if (mask & (1 << nxt)) continue;
                    long long add = c[last][nxt];               // adjacency cleaning cost
                    if (!single) add += e[prev][nxt];           // carry-over from two positions back
                    int nmask = mask | (1 << nxt);
                    long long &cell = dp[nmask][nxt * n + last]; // new last = nxt, new prev = last
                    if (cur + add < cell) cell = cur + add;
                }
            }
        }
    }

    cout << answer << "\n";
    return 0;
}
