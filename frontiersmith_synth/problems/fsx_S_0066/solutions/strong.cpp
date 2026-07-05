// TIER: strong
// Saturation-aware greedy + reassignment local search.
// Start from the greedy dispatch, then repeatedly move each forager to the reachable
// patch giving the best net marginal change (add - remove) until no improvement.
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

    vector<long long> load(P + 1, 0);
    vector<int> ans(B + 1, 0), appOf(B + 1, 0); // appOf[i] = appetite at ans[i]

    // ---- initial: saturation-aware greedy ----
    vector<int> order(B);
    for (int i = 0; i < B; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int x, int y){
        if (bestA[x] != bestA[y]) return bestA[x] > bestA[y];
        return x < y;
    });
    for (int oi = 0; oi < B; oi++) {
        int i = order[oi];
        long long bestGain = 0; int bestP = 0, bestApp = 0;
        for (auto& pr : reach[i]) {
            int j = pr.first, a = pr.second;
            long long room = S[j] - load[j];
            if (room < 0) room = 0;
            long long gain = min((long long)a, room);
            if (gain > bestGain) { bestGain = gain; bestP = j; bestApp = a; }
        }
        if (bestP != 0) { ans[i] = bestP; appOf[i] = bestApp; load[bestP] += bestApp; }
    }

    // ---- local search: reassign each forager to its best net patch ----
    auto contrib = [&](int j, long long l)->long long { return min(S[j], l); };
    for (int pass = 0; pass < 12; pass++) {
        bool improved = false;
        for (int i = 1; i <= B; i++) {
            int cur = ans[i];
            long long curApp = appOf[i];
            // remove i from current patch
            long long baseRemoved;
            if (cur != 0) {
                long long before = contrib(cur, load[cur]);
                long long after  = contrib(cur, load[cur] - curApp);
                baseRemoved = before - after; // value we lose if we leave cur
            } else baseRemoved = 0;

            // evaluate best target (including staying / idle)
            long long bestNet = 0;      // net over "remove then idle"
            int bestP = 0, bestApp = 0; // 0 = idle
            for (auto& pr : reach[i]) {
                int j = pr.first, a = pr.second;
                long long lj = (j == cur) ? load[j] - curApp : load[j];
                long long addGain = contrib(j, lj + a) - contrib(j, lj);
                if (addGain > bestNet) { bestNet = addGain; bestP = j; bestApp = a; }
            }

            // current net (keeping cur) relative to remove-then-idle baseline is baseRemoved
            long long curNet = baseRemoved;
            if (bestNet > curNet) {
                // apply move: remove from cur, add to bestP
                if (cur != 0) load[cur] -= curApp;
                if (bestP != 0) load[bestP] += bestApp;
                ans[i] = bestP; appOf[i] = (bestP == 0) ? 0 : bestApp;
                improved = true;
            }
        }
        if (!improved) break;
    }

    for (int i = 1; i <= B; i++) printf("%d%c", ans[i], i == B ? '\n' : ' ');
    return 0;
}
