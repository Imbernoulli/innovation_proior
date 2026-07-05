// TIER: strong
// Randomized multi-restart marginal-cost construction. Each restart processes segments
// in a random order; for each segment it repeatedly commits the night of least MARGINAL
// cost given the current global state -- free volunteers cost 0, contract costs
// p[t] plus D[t] ONLY when that night has no contract yet. This makes later segments
// cluster contract onto already-mobilized nights (amortizing D) and spends scarce
// volunteers where contract would be dearest. Keeps the cheapest schedule seen.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int T, K;
vector<int> vol, D, p, L, R, W;

int main() {
    if (scanf("%d %d", &T, &K) != 2) return 0;
    vol.assign(T + 1, 0); D.assign(T + 1, 0); p.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) scanf("%d", &vol[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &D[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &p[t]);
    L.assign(K + 1, 0); R.assign(K + 1, 0); W.assign(K + 1, 0);
    for (int i = 1; i <= K; i++) scanf("%d %d %d", &L[i], &R[i], &W[i]);

    mt19937 rng(20240707u);

    vector<vector<pair<int,int>>> bestAsg;
    ll bestCost = LLONG_MAX;

    int restarts = 24;
    for (int rs = 0; rs < restarts; rs++) {
        vector<int> remainVol = vol;
        vector<int> conCnt(T + 1, 0);              // contract guards per night so far
        vector<vector<pair<int,int>>> asg(K + 1);

        vector<int> ord(K);
        for (int i = 0; i < K; i++) ord[i] = i + 1;
        if (rs == 0) {
            // deterministic warm start: tightest windows first
            sort(ord.begin(), ord.end(), [&](int a, int b) {
                return (R[a] - L[a]) < (R[b] - L[b]);
            });
        } else {
            shuffle(ord.begin(), ord.end(), rng);
        }

        for (int i : ord) {
            int need = W[i];
            vector<char> used(T + 1, 0);
            while (need > 0) {
                int bestT = -1, bestMode = -1;
                ll bestMc = LLONG_MAX;
                for (int t = L[i]; t <= R[i]; t++) {
                    if (used[t]) continue;
                    // volunteer option
                    if (remainVol[t] > 0) {
                        ll mc = 0;
                        if (mc < bestMc) { bestMc = mc; bestT = t; bestMode = 0; }
                    }
                    // contract option
                    ll mc = p[t] + (conCnt[t] > 0 ? 0 : D[t]);
                    if (mc < bestMc) { bestMc = mc; bestT = t; bestMode = 1; }
                }
                if (bestT == -1) break; // window exhausted (shouldn't happen: W<=window)
                used[bestT] = 1;
                if (bestMode == 0) remainVol[bestT]--;
                else conCnt[bestT]++;
                asg[i].push_back({bestT, bestMode});
                need--;
            }
        }

        // evaluate cost
        vector<int> cc(T + 1, 0);
        for (int i = 1; i <= K; i++)
            for (auto& pr : asg[i]) if (pr.second == 1) cc[pr.first]++;
        ll cost = 0;
        for (int t = 1; t <= T; t++) {
            if (cc[t] > 0) cost += D[t];
            cost += (ll)cc[t] * p[t];
        }
        if (cost < bestCost) { bestCost = cost; bestAsg = asg; }
    }

    for (int i = 1; i <= K; i++) {
        printf("%d\n", (int)bestAsg[i].size());
        for (auto& pr : bestAsg[i]) printf("%d %d\n", pr.first, pr.second);
    }
    return 0;
}
