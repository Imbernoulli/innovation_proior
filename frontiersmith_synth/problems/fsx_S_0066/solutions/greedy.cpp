// TIER: greedy
// Saturation-aware one-pass greedy: process foragers by descending best appetite,
// send each to the reachable patch maximizing marginal gain clamp(S_j - load_j, 0, a).
#include <bits/stdc++.h>
using namespace std;
int main() {
    int P, B;
    scanf("%d %d", &P, &B);
    vector<long long> S(P + 1);
    for (int j = 1; j <= P; j++) scanf("%lld", &S[j]);

    vector<vector<pair<int,int>>> reach(B + 1);
    vector<int> bestA(B + 1, 0);
    for (int i = 1; i <= B; i++) {
        int m; scanf("%d", &m);
        reach[i].reserve(m);
        for (int k = 0; k < m; k++) {
            int p, a; scanf("%d %d", &p, &a);
            reach[i].push_back({p, a});
            bestA[i] = max(bestA[i], a);
        }
    }

    // order foragers by descending best appetite (tie: lower index first)
    vector<int> order(B);
    for (int i = 0; i < B; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int x, int y){
        if (bestA[x] != bestA[y]) return bestA[x] > bestA[y];
        return x < y;
    });

    vector<long long> load(P + 1, 0);
    vector<int> ans(B + 1, 0);
    for (int oi = 0; oi < B; oi++) {
        int i = order[oi];
        long long bestGain = -1; int bestP = 0;
        for (auto& pr : reach[i]) {
            int j = pr.first, a = pr.second;
            long long room = S[j] - load[j];
            if (room < 0) room = 0;
            long long gain = min((long long)a, room);
            if (gain > bestGain) { bestGain = gain; bestP = j; }
        }
        if (bestP != 0 && bestGain > 0) {
            ans[i] = bestP;
            // find appetite for bestP
            for (auto& pr : reach[i]) if (pr.first == bestP) { load[bestP] += pr.second; break; }
        } else {
            ans[i] = 0; // no positive marginal gain -> idle
        }
    }

    for (int i = 1; i <= B; i++) printf("%d%c", ans[i], i == B ? '\n' : ' ');
    return 0;
}
