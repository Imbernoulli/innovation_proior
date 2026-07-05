// TIER: greedy
// One-pass heuristic. Process segments tightest-window-first. For each segment grab
// free volunteers (earliest available nights) up to its need, then fill the rest with
// contract on the CHEAPEST per-guard-price nights of its window. Ignores mobilization
// sharing across segments -> leaves easy clustering gains on the table.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, K;
    if (scanf("%d %d", &T, &K) != 2) return 0;
    vector<int> vol(T + 1), D(T + 1), p(T + 1);
    for (int t = 1; t <= T; t++) scanf("%d", &vol[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &D[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &p[t]);
    vector<int> L(K + 1), R(K + 1), W(K + 1);
    for (int i = 1; i <= K; i++) scanf("%d %d %d", &L[i], &R[i], &W[i]);

    vector<int> remainVol = vol;

    // order segments by window length ascending (tightest first)
    vector<int> ord(K);
    for (int i = 0; i < K; i++) ord[i] = i + 1;
    sort(ord.begin(), ord.end(), [&](int a, int b) {
        return (R[a] - L[a]) < (R[b] - L[b]);
    });

    // assignments per segment: list of (night, mode)
    vector<vector<pair<int,int>>> asg(K + 1);

    for (int i : ord) {
        int need = W[i];
        vector<char> used(T + 1, 0);
        // volunteers first, earliest available nights
        for (int t = L[i]; t <= R[i] && need > 0; t++) {
            if (remainVol[t] > 0) {
                remainVol[t]--;
                used[t] = 1;
                asg[i].push_back({t, 0});
                need--;
            }
        }
        // remaining need -> contract on cheapest p[t] nights not yet used
        if (need > 0) {
            vector<int> cand;
            for (int t = L[i]; t <= R[i]; t++) if (!used[t]) cand.push_back(t);
            sort(cand.begin(), cand.end(), [&](int a, int b) { return p[a] < p[b]; });
            for (int idx = 0; idx < (int)cand.size() && need > 0; idx++) {
                asg[i].push_back({cand[idx], 1});
                need--;
            }
        }
    }

    for (int i = 1; i <= K; i++) {
        printf("%d\n", (int)asg[i].size());
        for (auto& pr : asg[i]) printf("%d %d\n", pr.first, pr.second);
    }
    return 0;
}
