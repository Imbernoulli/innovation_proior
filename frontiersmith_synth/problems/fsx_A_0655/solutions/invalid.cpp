// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Deliberately infeasible: build the first star cell (guaranteed degree >= 1)
// with a rotation that is OFF by one from the value its own incident edges
// require, i.e. it fails edge-angle-matching on its very first edge. Must
// score 0.

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
        if (i == 0 && type[0] == 0 && !angles[0].empty()){
            int rep = angles[0][0] % step0;
            int wrong = (rep + 1) % step0; // guaranteed to mismatch step0-consistent edges
            printf("1 %d %d\n", k0, wrong);
        } else if (type[i] == 0){
            int rep = angles[i].empty() ? 0 : (angles[i][0] % step0);
            bool ok = true;
            for (int a : angles[i]) if (a % step0 != rep) { ok = false; break; }
            if (ok) printf("1 %d %d\n", k0, rep);
            else printf("0 0 0\n");
        } else {
            printf("0 0 0\n");
        }
    }
    return 0;
}
