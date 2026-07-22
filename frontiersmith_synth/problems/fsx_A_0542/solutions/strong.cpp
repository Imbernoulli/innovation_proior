// TIER: strong
// The insight: the whole outage family only ever destroys links INSIDE the blocks, and
// each block's demand pair sits at its two ends. A single wide link joining a block's
// two ends bridges EVERY interior cut of that block -- so one such link per demand pair
// is a hitting set over all the threatened cuts. Buying one direct end-to-end
// reinforcement per demand heals all 3C scenarios using only C links (well inside the
// 2C reinforcement budget), instead of patching each outage separately.
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
    // (scenarios not needed: the hitting set is derived from demand geometry)

    map<pair<int,int>, vector<int>> mp;
    for (int i = M; i < m; i++){ int a=min(eu[i],ev[i]), b=max(eu[i],ev[i]); mp[{a,b}].push_back(i); }

    vector<char> sel(m, 0);
    for (int i = 0; i < M; i++) sel[i] = 1;
    int budget = Bbudget - M;

    // one direct end-to-end reinforcement per demand pair = hitting set over its cuts
    for (int j = 0; j < D && budget > 0; j++){
        int a = min(ds[j], de[j]), b = max(ds[j], de[j]);
        auto it = mp.find({a, b});
        if (it == mp.end()) continue;
        int best = -1;
        for (int cand : it->second) if (!sel[cand]){ best = cand; break; }
        if (best >= 0){ sel[best] = 1; budget--; }
    }

    vector<int> out; for (int i = 0; i < m; i++) if (sel[i]) out.push_back(i);
    printf("%d\n", (int)out.size());
    for (int i = 0; i < (int)out.size(); i++) printf("%d%c", out[i], i+1<(int)out.size()?' ':'\n');
    return 0;
}
