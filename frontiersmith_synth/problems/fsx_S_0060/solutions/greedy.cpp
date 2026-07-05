// TIER: greedy
// Per-well: pick the R_j cheapest steps in [1,d_j], using spot where capacity
// remains (and spot is not dearer than on-demand), else on-demand. Wells handled
// in deadline order so tight wells grab slots first. Ignores contiguity/warm-up.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, T; long long warm;
    scanf("%d %d %lld", &N, &T, &warm);
    vector<long long> od(T + 1), sp(T + 1);
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

    for (int idx = 0; idx < N; idx++) {
        int j = order[idx];
        // candidate steps 1..d_j with an effective (achievable) cost right now
        vector<pair<long long,int>> cand; // (effective cost, step)
        cand.reserve(d[j]);
        for (int t = 1; t <= d[j]; t++) {
            long long eff = od[t];
            if (capleft[t] > 0 && sp[t] < eff) eff = sp[t];
            cand.push_back({eff, t});
        }
        sort(cand.begin(), cand.end());
        int need = R[j];
        for (int c = 0; c < (int)cand.size() && need > 0; c++) {
            int t = cand[c].second;
            if (capleft[t] > 0 && sp[t] <= od[t]) {
                mode[j][t] = 1; capleft[t]--;
            } else {
                mode[j][t] = 2;
            }
            need--;
        }
    }

    for (int j = 1; j <= N; j++)
        for (int t = 1; t <= T; t++)
            printf("%d%c", mode[j][t], t == T ? '\n' : ' ');
    return 0;
}
