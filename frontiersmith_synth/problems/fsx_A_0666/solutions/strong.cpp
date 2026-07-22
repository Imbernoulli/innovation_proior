// TIER: strong
// Insight: interference is additive, so admission must be checked against the
// TRUE cumulative sum, and every accepted link should use the MINIMUM power
// that still clears its own threshold (not the max) -- a weaker transmitter
// imposes less interference on everyone sharing its channel, which is exactly
// what buys extra spatial reuse. Process links value-first; for each, try
// every (channel, power) pair (power ascending) and accept the first one that
// (a) lets the link itself clear theta given already-active same-channel
// links, and (b) does not push any already-active same-channel link below
// theta. Skip the link if nothing works.
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

    vector<int> outc(N + 1, -1), outk(N + 1, -1);

    vector<int> idx(N);
    for (int i = 0; i < N; i++) idx[i] = i + 1;
    stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return w[a] > w[b]; });

    vector<vector<int>> byChannel(C + 1); // active links per channel

    for (int link : idx) {
        int bestC = -1, bestK = -1;
        for (int c = 1; c <= C && bestC == -1; c++) {
            for (int k = 1; k <= K; k++) {
                double sig = (double)P[k] * gain(tx[link], ty[link], rx[link], ry[link]);
                double inter = (double)N0;
                for (int j : byChannel[c])
                    inter += (double)P[outk[j]] * gain(tx[j], ty[j], rx[link], ry[link]);
                if (sig / inter < theta - 1e-9) continue; // this link itself fails

                // does adding `link` at power k break any currently active
                // link j on channel c?
                bool ok = true;
                for (int j : byChannel[c]) {
                    double sig_j = (double)P[outk[j]] * gain(tx[j], ty[j], rx[j], ry[j]);
                    double inter_j = (double)N0;
                    for (int j2 : byChannel[c]) {
                        if (j2 == j) continue;
                        inter_j += (double)P[outk[j2]] * gain(tx[j2], ty[j2], rx[j], ry[j]);
                    }
                    inter_j += (double)P[k] * gain(tx[link], ty[link], rx[j], ry[j]);
                    if (sig_j / inter_j < theta - 1e-9) { ok = false; break; }
                }
                if (ok) { bestC = c; bestK = k; break; }
            }
        }
        if (bestC != -1) {
            outc[link] = bestC;
            outk[link] = bestK;
            byChannel[bestC].push_back(link);
        }
    }

    for (int i = 1; i <= N; i++) printf("%d %d\n", outc[i], outk[i]);
    return 0;
}
