// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Build every star cell with the single fixed order palette[0] (always
// self-consistent, since every star cell's own edges share one angle by
// construction). Never build any adapter cell. This reproduces the checker's
// internal baseline B exactly.

int main(){
    int N, M;
    scanf("%d %d", &N, &M);
    vector<int> type(N);
    for (int i = 0; i < N; i++) scanf("%d", &type[i]);
    int p; scanf("%d", &p);
    vector<int> paletteK(p), paletteC(p);
    for (int i = 0; i < p; i++) scanf("%d %d", &paletteK[i], &paletteC[i]);
    ll W1, W2, W3; scanf("%lld %lld %lld", &W1, &W2, &W3);
    vector<int> cost(N);
    for (int i = 0; i < N; i++) scanf("%d", &cost[i]);
    vector<vector<int>> angles(N);
    vector<int> eu(M), ev(M), ea(M), elen(M);
    for (int i = 0; i < M; i++){
        scanf("%d %d %d %d", &eu[i], &ev[i], &ea[i], &elen[i]);
        angles[eu[i]].push_back(ea[i]);
        angles[ev[i]].push_back(ea[i]);
    }

    const int L = 24;
    int k0 = paletteK[0];
    int step0 = L / k0;

    for (int i = 0; i < N; i++){
        if (type[i] != 0){ printf("0 0 0\n"); continue; }
        int rep = angles[i].empty() ? 0 : (angles[i][0] % step0);
        bool ok = true;
        for (int a : angles[i]) if (a % step0 != rep) { ok = false; break; }
        if (ok) printf("1 %d %d\n", k0, rep);
        else printf("0 0 0\n");
    }
    return 0;
}
