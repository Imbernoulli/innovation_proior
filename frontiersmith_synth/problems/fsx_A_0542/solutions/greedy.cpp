// TIER: greedy
// The obvious "add a backup for every failed link" heuristic: keep the spine, then
// walk the outage scenarios in order and, for each destroyed link, add the cheapest
// spare parallel to it (a narrow spare). With budget = M + 2C this patches only 2C of
// the 3C scenarios and leaves C severed -- it never realizes one wide link heals a
// whole block's three scenarios at once.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    int n, m, M, K, D, Bbudget; ll P;
    if (scanf("%d %d %d %d %d %d %lld", &n, &m, &M, &K, &D, &Bbudget, &P) != 7) return 0;
    vector<int> eu(m), ev(m); vector<ll> ew(m);
    for (int i = 0; i < m; i++){ ll u, v, w; scanf("%lld %lld %lld", &u, &v, &w); eu[i]=u; ev[i]=v; ew[i]=w; }
    vector<int> ds(D), de(D);
    for (int j = 0; j < D; j++) scanf("%d %d", &ds[j], &de[j]);
    vector<vector<int>> scen(K);
    for (int k = 0; k < K; k++){ int c; scanf("%d", &c); scen[k].resize(c); for (int t=0;t<c;t++) scanf("%d", &scen[k][t]); }

    // endpoint-pair -> reinforcement (non-spine) links
    map<pair<int,int>, vector<int>> mp;
    for (int i = M; i < m; i++){ int a=min(eu[i],ev[i]), b=max(eu[i],ev[i]); mp[{a,b}].push_back(i); }

    vector<char> sel(m, 0);
    for (int i = 0; i < M; i++) sel[i] = 1;
    int budget = Bbudget - M;

    for (int k = 0; k < K && budget > 0; k++){
        for (int d : scen[k]){
            if (budget <= 0) break;
            int a = min(eu[d], ev[d]), b = max(eu[d], ev[d]);
            auto it = mp.find({a, b});
            if (it == mp.end()) continue;
            int best = -1; ll bw = LLONG_MAX;                 // cheapest spare = the narrow copy
            for (int cand : it->second) if (!sel[cand] && ew[cand] < bw){ bw = ew[cand]; best = cand; }
            if (best >= 0){ sel[best] = 1; budget--; }
        }
    }

    vector<int> out; for (int i = 0; i < m; i++) if (sel[i]) out.push_back(i);
    printf("%d\n", (int)out.size());
    for (int i = 0; i < (int)out.size(); i++) printf("%d%c", out[i], i+1<(int)out.size()?' ':'\n');
    return 0;
}
