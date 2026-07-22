// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Obvious first attempt: every star cell uses the same fixed order palette[0]
// (no diversity insight) -- BUT since adapter cells clearly need to bridge
// two different angles, "just use the safest, most general motif" -- always
// build every adapter with the universal order k=24 (step=1 matches any
// angle unconditionally), regardless of how much cheaper a smaller order
// would have sufficed. This recovers the adapter-crossing strand length but
// systematically overpays the adapter cost budget.

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
        if (type[i] == 0){
            int rep = angles[i].empty() ? 0 : (angles[i][0] % step0);
            bool ok = true;
            for (int a : angles[i]) if (a % step0 != rep) { ok = false; break; }
            if (ok) printf("1 %d %d\n", k0, rep);
            else printf("0 0 0\n");
        } else {
            // universal adapter: k=24, step=1, matches any angle, r arbitrary
            printf("1 24 0\n");
        }
    }
    return 0;
}
