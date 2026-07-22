// TIER: invalid
// Deliberately infeasible: boosts a pipe that is NOT booster-ready (cand=0) --
// or, if every pipe happens to be booster-ready (never in this generator),
// falls back to breaking flow conservation outright. Must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int V, E, S, T, K;
    scanf("%d %d %d %d %d", &V, &E, &S, &T, &K);
    vector<int> sn(S); vector<ll> sa(S);
    for (int i = 0; i < S; i++) scanf("%d %lld", &sn[i], &sa[i]);
    vector<int> tn(T); vector<ll> ta(T);
    for (int i = 0; i < T; i++) scanf("%d %lld", &tn[i], &ta[i]);

    vector<int> eu(E), ev(E), ecand(E);
    vector<ll> er(E), egain(E);
    int nonCand = -1;
    for (int e = 0; e < E; e++) {
        scanf("%d %d %lld %d %lld", &eu[e], &ev[e], &er[e], &ecand[e], &egain[e]);
        if (ecand[e] == 0 && nonCand < 0) nonCand = e;
    }

    if (nonCand >= 0 && K >= 1) {
        // boost a non-boostable pipe -- checker must reject with quitf.
        printf("1 %d", nonCand + 1);
        for (int e = 0; e < E; e++) printf(" %.6f", 0.0);
        printf("\n");
    } else {
        // fallback: dump all of the first source's requirement onto pipe 1 in the
        // WRONG direction relative to node balance, breaking conservation.
        printf("0");
        for (int e = 0; e < E; e++) {
            double v = (e == 0) ? (double)(1 + sa[0]) : 0.0;
            printf(" %.6f", v);
        }
        printf("\n");
    }
    return 0;
}
