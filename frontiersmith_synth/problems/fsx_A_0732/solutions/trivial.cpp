// TIER: trivial
// Ignore value AND structure: walk arenas in ascending (lowest-available-id)
// order and take each arena's HIGHEST-id available node (structurally always
// a market node, never a control -- generation order lists controls first),
// one per arena, until the budget is spent. This is exactly the checker's
// internal baseline construction -- a "grab whatever comes first" default
// with no attempt at picking good nodes, good arenas, or good PAIRS.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int par[400005];
int find(int x){ return par[x] == x ? x : par[x] = find(par[x]); }
void uni(int a, int b){ a = find(a); b = find(b); if (a != b) par[a] = b; }

int main(){
    int N, M, K, S;
    scanf("%d %d %d %d", &N, &M, &K, &S);
    vector<ll> val(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &val[i]);
    for (int i = 1; i <= N; i++) par[i] = i;
    for (int i = 0; i < M; i++){ int u, v; scanf("%d %d", &u, &v); uni(u, v); }
    vector<char> isRival(N + 1, 0);
    for (int i = 0; i < S; i++){ int b; scanf("%d", &b); isRival[b] = 1; }

    map<int, int> lowest, highest; // component root -> lowest / highest available id in it
    for (int v = 1; v <= N; v++){
        if (isRival[v]) continue;
        int r = find(v);
        auto itl = lowest.find(r);
        if (itl == lowest.end() || v < itl->second) lowest[r] = v;
        auto ith = highest.find(r);
        if (ith == highest.end() || v > ith->second) highest[r] = v;
    }
    // order arenas by ascending lowest-available-id, but seed each arena's
    // HIGHEST-id available node (see header comment)
    vector<pair<int,int>> arenaOrder;   // (lowest id for ordering, highest id to seed)
    for (auto &kv : lowest) arenaOrder.push_back({kv.second, highest[kv.first]});
    sort(arenaOrder.begin(), arenaOrder.end());

    for (int i = 0; i < K && i < (int)arenaOrder.size(); i++) printf("%d\n", arenaOrder[i].second);
    return 0;
}
