// TIER: greedy
// Earliest-start non-delay list scheduling: repeatedly dispatch the ready stop
// with the smallest earliest start time (tie-break: shortest green window, then convoy id).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int M, J;
    if (scanf("%d %d", &M, &J) != 2) return 0;
    vector<int> L(J);
    vector<vector<int>> mach(J), p(J);
    ll total = 0;
    for (int j = 0; j < J; j++) {
        scanf("%d", &L[j]);
        mach[j].resize(L[j]);
        p[j].resize(L[j]);
        for (int i = 0; i < L[j]; i++) { scanf("%d %d", &mach[j][i], &p[j][i]); total++; }
    }
    vector<ll> machFree(M + 1, 0), jobReady(J, 0);
    vector<int> nxt(J, 0);
    vector<vector<ll>> st(J);
    for (int j = 0; j < J; j++) st[j].assign(L[j], 0);

    ll done = 0;
    while (done < total) {
        int bj = -1; ll bestEst = 0; int bp = 0;
        for (int j = 0; j < J; j++) {
            if (nxt[j] >= L[j]) continue;
            int i = nxt[j], m = mach[j][i], pp = p[j][i];
            ll est = max(jobReady[j], machFree[m]);
            if (bj == -1 || est < bestEst ||
                (est == bestEst && pp < bp) ||
                (est == bestEst && pp == bp && j < bj)) {
                bj = j; bestEst = est; bp = pp;
            }
        }
        int i = nxt[bj], m = mach[bj][i];
        st[bj][i] = bestEst;
        machFree[m] = bestEst + p[bj][i];
        jobReady[bj] = bestEst + p[bj][i];
        nxt[bj]++;
        done++;
    }
    for (int j = 0; j < J; j++) {
        for (int i = 0; i < L[j]; i++) printf("%lld ", st[j][i]);
        printf("\n");
    }
    return 0;
}
