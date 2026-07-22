// TIER: strong
// Insight: the 40 ducting scenarios are ONE sliding window over separation
// distance, not 40 unrelated graphs. Encode each station's corridor coordinate
// into its channel: c_i = 1 + (floor((x_i+phase)/W) mod C). The channel then
// repeats in SPACE with period roughly W*C -- search a ladder of block widths W
// (and a few phases) so that period is pushed OUTSIDE the union of all 40 swept
// ducting bands, escaping every scenario with one formula instead of solving 40
// separate colorings. A local min-conflict repair pass (restricted to base
// edges only) then cleans up any residual local collision inside a block. Every
// candidate is scored EXACTLY like the checker (max over base+scenario cost) on
// the real input edges, and the best candidate found is kept.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, C, K;
    if (!(cin >> N >> C >> K)) return 0;
    vector<ll> x(N + 1);
    for (int i = 1; i <= N; i++) cin >> x[i];

    int M0; cin >> M0;
    vector<int> bI(M0), bJ(M0); vector<ll> bW(M0);
    vector<vector<pair<int,ll>>> adj(N + 1);
    for (int e = 0; e < M0; e++) {
        int i, j; ll w; cin >> i >> j >> w;
        bI[e] = i; bJ[e] = j; bW[e] = w;
        adj[i].push_back({j, w});
        adj[j].push_back({i, w});
    }

    vector<vector<int>> dI(K + 1), dJ(K + 1);
    vector<vector<ll>> dW(K + 1);
    for (int s = 1; s <= K; s++) {
        int Ms; cin >> Ms;
        dI[s].resize(Ms); dJ[s].resize(Ms); dW[s].resize(Ms);
        for (int e = 0; e < Ms; e++) cin >> dI[s][e] >> dJ[s][e] >> dW[s][e];
    }

    auto evalF = [&](const vector<int>& c) -> ll {
        ll baseConflict = 0;
        for (int e = 0; e < M0; e++) if (c[bI[e]] == c[bJ[e]]) baseConflict += bW[e];
        ll F = 0;
        for (int s = 1; s <= K; s++) {
            ll dc = 0;
            for (size_t e = 0; e < dI[s].size(); e++)
                if (c[dI[s][e]] == c[dJ[s][e]]) dc += dW[s][e];
            F = max(F, baseConflict + dc);
        }
        return F;
    };

    vector<ll> Wlist = {1,2,3,4,5,6,8,10,13,17,22,28,36,46,60,78,100,130,170,220,290,380,500,650,850,1100};
    vector<ll> phases = {0, 1, 2};

    vector<int> best(N + 1, 1);
    ll bestF = LLONG_MAX;
    vector<int> cand(N + 1);
    vector<ll> confl(C + 1);

    for (ll W : Wlist) {
        for (ll ph : phases) {
            for (int i = 1; i <= N; i++) {
                ll block = (x[i] + ph) / W;
                ll ch = ((block % C) + C) % C;
                cand[i] = (int)ch + 1;
            }
            // local min-conflict repair against already-processed base neighbours
            for (int j = 1; j <= N; j++) {
                fill(confl.begin(), confl.end(), 0);
                for (auto &pr : adj[j]) {
                    int nb = pr.first; ll w = pr.second;
                    if (nb < j) confl[cand[nb]] += w;
                }
                int curCh = cand[j];
                if (confl[curCh] == 0) continue;
                int bestCh = curCh; ll bestConf = confl[curCh];
                for (int c = 1; c <= C; c++) {
                    if (confl[c] < bestConf ||
                        (confl[c] == bestConf && abs(c - curCh) < abs(bestCh - curCh))) {
                        bestConf = confl[c]; bestCh = c;
                    }
                }
                cand[j] = bestCh;
            }
            ll F = evalF(cand);
            if (F < bestF) { bestF = F; best = cand; }
        }
    }

    for (int i = 1; i <= N; i++) cout << best[i] << (i < N ? ' ' : '\n');
    return 0;
}
