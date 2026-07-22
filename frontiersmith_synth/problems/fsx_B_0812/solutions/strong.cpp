// TIER: strong
// The insight: booths are relocatable (conserved per pipeline, not fixed in place).
// For each pipeline, figure out how many booths a CHEAPEST-everywhere chunking would
// actually need to survive the clock; if the pipeline's own conserved booth total
// covers that, relocate the booths to exactly those chunk boundaries (dumping any
// spare booths on the cheapest-relocation belt) and run almost the whole pipeline on
// its cheap variant. Only when a pipeline is genuinely booth-starved does it fall back
// to the fixed-placement upsizing recipe.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Variant { ll d, a; };

int main() {
    int N, K; ll T;
    scanf("%d %d %lld", &N, &K, &T);
    vector<int> lens(K);
    for (int p = 0; p < K; p++) scanf("%d", &lens[p]);

    vector<int> Kv(N + 1);
    vector<array<Variant,3>> menu(N + 1);
    for (int v = 1; v <= N; v++) {
        int kv; scanf("%d", &kv); Kv[v] = kv;
        for (int i = 0; i < kv; i++) {
            ll d, a; scanf("%lld %lld", &d, &a);
            menu[v][i] = {d, a};
        }
    }

    vector<vector<ll>> w(K), cst(K);
    for (int p = 0; p < K; p++) {
        int L = lens[p];
        w[p].assign(L - 1, 0);
        cst[p].assign(L - 1, 0);
        for (int i = 0; i < L - 1; i++) scanf("%lld %lld", &w[p][i], &cst[p][i]);
    }

    vector<int> off(K);
    { int cur = 0; for (int p = 0; p < K; p++) { off[p] = cur; cur += lens[p]; } }

    vector<int> x(N + 1, 0);
    vector<vector<ll>> wOut(K);

    for (int p = 0; p < K; p++) {
        int L = lens[p];
        // baseline: the cheapest variant that still fits under T alone (defensive;
        // by construction the cheapest variant already fits for every station here)
        vector<int> effX(L);
        vector<ll> effD(L);
        for (int i = 0; i < L; i++) {
            int v = off[p] + i + 1;
            int k = 0;
            while (k < Kv[v] - 1 && menu[v][k].d > T) k++;
            effX[i] = k; effD[i] = menu[v][k].d;
        }

        ll Wtot = 0; for (int i = 0; i < L - 1; i++) Wtot += w[p][i];

        // chunk cheapest-everywhere against T, recording required break belts
        vector<int> breakBelt;
        ll run = 0;
        for (int i = 0; i < L; i++) {
            ll d = effD[i];
            if (run > 0 && run + d > T) { breakBelt.push_back(i - 1); run = d; }
            else run += d;
        }
        ll needed = (ll)breakBelt.size();

        wOut[p].assign(L - 1, 0);

        if (needed <= Wtot) {
            for (int idx : breakBelt) wOut[p][idx] = 1;
            ll spare = Wtot - needed;
            if (spare > 0 && L - 1 > 0) {
                int best = 0;
                for (int i = 1; i < L - 1; i++) if (cst[p][i] < cst[p][best]) best = i;
                wOut[p][best] += spare;
            }
            for (int i = 0; i < L; i++) x[off[p] + i + 1] = effX[i];
        } else {
            // booth-starved: fall back to fixed original placement + greedy repair
            for (int i = 0; i < L - 1; i++) wOut[p][i] = w[p][i];
            for (int i = 0; i < L; i++) x[off[p] + i + 1] = effX[i];
            int start = 0;
            for (int i = 0; i < L; i++) {
                bool freshRun = (i == 0) || (w[p][i - 1] >= 1);
                if (freshRun) start = i;
                bool checkpoint = (i == L - 1) || (i < L - 1 && w[p][i] >= 1);
                if (!checkpoint) continue;
                while (true) {
                    ll r = 0;
                    for (int j = start; j <= i; j++) r += menu[off[p] + j + 1][x[off[p] + j + 1]].d;
                    if (r <= T) break;
                    int best = -1; ll bestDelay = -1;
                    for (int j = start; j <= i; j++) {
                        int v = off[p] + j + 1;
                        if (x[v] < Kv[v] - 1) {
                            ll dd = menu[v][x[v]].d;
                            if (dd > bestDelay) { bestDelay = dd; best = v; }
                        }
                    }
                    if (best == -1) break;
                    x[best]++;
                }
            }
        }
    }

    for (int v = 1; v <= N; v++) printf("%d ", x[v]);
    printf("\n");
    for (int p = 0; p < K; p++)
        for (int i = 0; i < (int)wOut[p].size(); i++) printf("%lld ", wOut[p][i]);
    printf("\n");
    return 0;
}
