// TIER: strong
// Aperture sweep: for every menu aperture, sort clusters by v_i(r) descending and
// grid-greedily pack a collision-free independent set (deploy only v>0), then add the
// band-diversity bonus. Keep the aperture giving the best total F. Finds the sweet-spot
// aperture (not the largest) and the planted hidden grid.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K, NB; long long D;
    if (scanf("%d %d %lld %d", &N, &K, &D, &NB) != 4) return 0;
    vector<long long> R(K);
    for (int t = 0; t < K; t++) scanf("%lld", &R[t]);
    vector<long long> X(N), Y(N), W(N), C(N), A(N);
    vector<int> Bd(N);
    for (int i = 0; i < N; i++)
        scanf("%lld %lld %lld %lld %lld %d", &X[i], &Y[i], &W[i], &C[i], &A[i], &Bd[i]);

    long long bestF = LLONG_MIN;
    int bestT = 0;
    vector<int> bestSel;

    vector<int> ord(N);
    for (int t = 0; t < K; t++) {
        long long r = R[t], cell = 2 * r, thr = 4 * r * r;
        vector<long long> v(N);
        for (int i = 0; i < N; i++) v[i] = W[i] * r - C[i] * r * r - A[i];
        for (int i = 0; i < N; i++) ord[i] = i;
        sort(ord.begin(), ord.end(), [&](int a, int b) { return v[a] > v[b]; });

        unordered_map<long long, vector<int>> grid;
        grid.reserve(N * 2);
        auto key = [&](long long a, long long b) { return a * 4000003LL + b; };
        vector<int> sel;
        set<int> bands;
        long long sumv = 0;
        for (int idx = 0; idx < N; idx++) {
            int i = ord[idx];
            if (v[i] <= 0) break;                       // rest are non-positive
            long long cx = X[i] / cell, cy = Y[i] / cell;
            bool ok = true;
            for (long long dx = -1; dx <= 1 && ok; dx++)
                for (long long dy = -1; dy <= 1 && ok; dy++) {
                    auto it = grid.find(key(cx + dx, cy + dy));
                    if (it == grid.end()) continue;
                    for (int j : it->second) {
                        long long ddx = X[i] - X[j], ddy = Y[i] - Y[j];
                        if (ddx * ddx + ddy * ddy < thr) { ok = false; break; }
                    }
                }
            if (!ok) continue;
            grid[key(cx, cy)].push_back(i);
            sel.push_back(i);
            sumv += v[i];
            bands.insert(Bd[i]);
        }
        long long F = sumv + D * (long long)bands.size();
        if (F > bestF) { bestF = F; bestT = t; bestSel = sel; }
    }

    if (bestF == LLONG_MIN) { bestT = 0; bestSel.clear(); }
    printf("%d %d\n", bestT + 1, (int)bestSel.size());
    for (size_t k = 0; k < bestSel.size(); k++)
        printf("%d%c", bestSel[k] + 1, k + 1 < bestSel.size() ? ' ' : '\n');
    if (bestSel.empty()) printf("\n");
    return 0;
}
