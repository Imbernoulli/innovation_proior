// TIER: strong
// Wells handled in deadline order (tight wells reserve scarce spot first). For each
// well, given the per-step achievable price (spot where capacity remains and cheaper,
// else on-demand), a DP picks exactly R_j active steps in [1,d_j] minimizing
//   sum(step price) + warm * (number of maximal active runs).
// This is warm-aware and per-step optimal, so it dominates greedy's contiguity-blind
// cheapest-R_j selection.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF = (ll)1e18;

int main() {
    int N, T; ll warm;
    scanf("%d %d %lld", &N, &T, &warm);
    vector<ll> od(T + 1), sp(T + 1);
    vector<int> C(T + 1), capleft(T + 1);
    for (int t = 1; t <= T; t++) scanf("%lld", &od[t]);
    for (int t = 1; t <= T; t++) scanf("%lld", &sp[t]);
    for (int t = 1; t <= T; t++) { scanf("%d", &C[t]); capleft[t] = C[t]; }
    vector<int> R(N + 1), d(N + 1);
    for (int j = 1; j <= N; j++) scanf("%d %d", &R[j], &d[j]);

    vector<vector<int>> mode(N + 1, vector<int>(T + 1, 0));

    vector<int> order(N);
    for (int j = 0; j < N; j++) order[j] = j + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (d[a] != d[b]) return d[a] < d[b];
        return R[a] > R[b];
    });

    for (int oi = 0; oi < N; oi++) {
        int j = order[oi];
        int L = d[j], K = R[j];

        // achievable price + whether spot is the achievable choice at each step
        vector<ll> c(L + 1);
        vector<char> spotOK(L + 1, 0);
        for (int t = 1; t <= L; t++) {
            if (capleft[t] > 0 && sp[t] <= od[t]) { c[t] = sp[t]; spotOK[t] = 1; }
            else { c[t] = od[t]; spotOK[t] = 0; }
        }

        // DP[s][k][p]: min cost after processing steps 1..s, k active chosen,
        // p = whether step s was active (p=1) or not (p=0).
        vector<vector<array<ll,2>>> DP(L + 1, vector<array<ll,2>>(K + 1, {INF, INF}));
        DP[0][0][0] = 0; // before any step: 0 chosen, "previous" inactive
        for (int s = 1; s <= L; s++) {
            for (int k = 0; k <= K; k++) {
                // skip step s -> ends inactive
                DP[s][k][0] = min(DP[s-1][k][0], DP[s-1][k][1]);
                // choose step s -> ends active
                if (k >= 1) {
                    ll fromInact = (DP[s-1][k-1][0] == INF) ? INF : DP[s-1][k-1][0] + c[s] + warm;
                    ll fromAct   = (DP[s-1][k-1][1] == INF) ? INF : DP[s-1][k-1][1] + c[s];
                    DP[s][k][1] = min(fromInact, fromAct);
                }
            }
        }

        // reconstruct chosen active steps
        vector<char> act(L + 1, 0);
        int pp = (DP[L][K][0] <= DP[L][K][1]) ? 0 : 1;
        int kk = K;
        for (int s = L; s >= 1; s--) {
            if (pp == 1) {
                act[s] = 1;
                ll fromInact = (DP[s-1][kk-1][0] == INF) ? INF : DP[s-1][kk-1][0] + c[s] + warm;
                ll fromAct   = (DP[s-1][kk-1][1] == INF) ? INF : DP[s-1][kk-1][1] + c[s];
                kk -= 1;
                pp = (fromInact <= fromAct) ? 0 : 1;
            } else {
                pp = (DP[s-1][kk][0] <= DP[s-1][kk][1]) ? 0 : 1;
            }
        }

        // commit: spot where achievable, else on-demand; decrement capacity
        for (int t = 1; t <= L; t++) {
            if (!act[t]) continue;
            if (spotOK[t] && capleft[t] > 0) { mode[j][t] = 1; capleft[t]--; }
            else { mode[j][t] = 2; }
        }
    }

    for (int j = 1; j <= N; j++)
        for (int t = 1; t <= T; t++)
            printf("%d%c", mode[j][t], t == T ? '\n' : ' ');
    return 0;
}
