// TIER: greedy
// The "obvious" approach: build a PAIRWISE conflict graph (two links conflict
// if, at max power with only the OTHER as sole interferer, either one's SINR
// would fall below theta), then greedily colour it with C colours and turn
// EVERY link ON at max power. This ignores that interference is additive: a
// cluster of links that are all pairwise fine together can still collectively
// drown each other once several of them share a channel simultaneously.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, C, K;
ll alphaExp;
double theta;
ll N0;
vector<ll> P;
vector<ll> tx, ty, rx, ry, w;

double gain(ll ax, ll ay, ll bx, ll by) {
    ll dx = ax - bx, dy = ay - by;
    ll D = dx * dx + dy * dy;
    if (D < 1) D = 1;
    double Dd = (double)D, denom = 1.0;
    int m = (int)(alphaExp / 2);
    for (int t = 0; t < m; t++) denom *= Dd;
    return 1.0 / denom;
}

int main() {
    scanf("%d %d %d", &N, &C, &K);
    scanf("%lld %lf %lld", &alphaExp, &theta, &N0);
    P.assign(K + 1, 0);
    for (int j = 1; j <= K; j++) scanf("%lld", &P[j]);
    tx.assign(N + 1, 0); ty.assign(N + 1, 0); rx.assign(N + 1, 0); ry.assign(N + 1, 0); w.assign(N + 1, 0);
    for (int i = 1; i <= N; i++)
        scanf("%lld %lld %lld %lld %lld", &tx[i], &ty[i], &rx[i], &ry[i], &w[i]);

    // Pairwise conflict test at max power P[K].
    vector<vector<int>> conflict(N + 1);
    for (int i = 1; i <= N; i++) {
        double sig_i = (double)P[K] * gain(tx[i], ty[i], rx[i], ry[i]);
        for (int j = 1; j <= N; j++) {
            if (i == j) continue;
            double inter_i = (double)N0 + (double)P[K] * gain(tx[j], ty[j], rx[i], ry[i]);
            if (sig_i / inter_i < theta) { conflict[i].push_back(j); }
        }
    }

    vector<int> idx(N);
    for (int i = 0; i < N; i++) idx[i] = i + 1;
    stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return w[a] > w[b]; });

    vector<int> outc(N + 1, 0);
    for (int link : idx) {
        vector<bool> usedByNeighbor(C + 1, false);
        for (int nb : conflict[link])
            if (outc[nb] != 0) usedByNeighbor[outc[nb]] = true;
        int chosen = -1;
        for (int c = 1; c <= C; c++)
            if (!usedByNeighbor[c]) { chosen = c; break; }
        if (chosen == -1) chosen = 1; // naive fallback: forced collision
        outc[link] = chosen;
    }
    for (int i = 1; i <= N; i++) printf("%d %d\n", outc[i], K); // always max power
    return 0;
}
