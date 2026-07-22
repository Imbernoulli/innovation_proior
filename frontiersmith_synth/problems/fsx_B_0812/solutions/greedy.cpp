// TIER: greedy
// The "obvious" recipe: never touch the booth (register) placement -- just start
// every station at its cheapest variant and, whenever a combinational run would
// blow the clock, repeatedly upsize the single largest-delay station in that run
// by one variant step until it fits. No retiming = the trap.
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

    vector<vector<ll>> w(K);
    for (int p = 0; p < K; p++) {
        int L = lens[p];
        w[p].assign(L - 1, 0);
        for (int i = 0; i < L - 1; i++) {
            ll c; scanf("%lld %lld", &w[p][i], &c);
        }
    }

    vector<int> off(K);
    { int cur = 0; for (int p = 0; p < K; p++) { off[p] = cur; cur += lens[p]; } }

    vector<int> x(N + 1, 0); // start everyone cheapest

    for (int p = 0; p < K; p++) {
        int L = lens[p];
        int start = 0;
        for (int i = 0; i < L; i++) {
            bool freshRun = (i == 0) || (w[p][i - 1] >= 1);
            if (freshRun) start = i;
            bool checkpoint = (i == L - 1) || (i < L - 1 && w[p][i] >= 1);
            if (!checkpoint) continue;

            // recompute the run [start..i]'s arrival, repair while it exceeds T
            while (true) {
                ll run = 0;
                for (int j = start; j <= i; j++) run += menu[off[p] + j + 1][x[off[p] + j + 1]].d;
                if (run <= T) break;
                // find the station in [start,i] with the largest CURRENT delay that can still be upsized
                int best = -1; ll bestDelay = -1;
                for (int j = start; j <= i; j++) {
                    int v = off[p] + j + 1;
                    if (x[v] < Kv[v] - 1) {
                        ll dd = menu[v][x[v]].d;
                        if (dd > bestDelay) { bestDelay = dd; best = v; }
                    }
                }
                if (best == -1) break; // everyone already fastest (guaranteed feasible by construction)
                x[best]++;
            }
        }
    }

    for (int v = 1; v <= N; v++) printf("%d ", x[v]);
    printf("\n");
    // booth placement unchanged
    for (int p = 0; p < K; p++)
        for (int i = 0; i < (int)w[p].size(); i++) printf("%lld ", w[p][i]);
    printf("\n");
    return 0;
}
