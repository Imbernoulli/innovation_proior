// TIER: greedy
// Fixed middle aperture, scan clusters in INPUT ORDER, deploy if v>0 and collision-free
// (grid-checked). Ignores value ranking, aperture tuning, and the band bonus. Weak but
// feasible; diverges from trivial and strong.
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

    int t = K / 2;                         // middle aperture
    long long r = R[t], cell = 2 * r, thr = 4 * r * r;
    unordered_map<long long, vector<int>> grid;
    auto key = [&](long long a, long long b) { return a * 4000003LL + b; };
    vector<int> sel;
    for (int i = 0; i < N; i++) {
        long long v = W[i] * r - C[i] * r * r - A[i];
        if (v <= 0) continue;
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
    }
    printf("%d %d\n", t + 1, (int)sel.size());
    for (size_t k = 0; k < sel.size(); k++) printf("%d%c", sel[k] + 1, k + 1 < sel.size() ? ' ' : '\n');
    if (sel.empty()) printf("\n");
    return 0;
}
