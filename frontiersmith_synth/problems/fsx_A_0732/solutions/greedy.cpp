// TIER: greedy
// The obvious "influence maximisation" move: place ONE strong counter-seed in as
// many distinct arenas as possible (spread thin, mirror the rival's one-hub-per-
// arena footprint), only doubling up once every arena already has a seed. This
// never intentionally builds an adjacent pair, so -- per the majority-threshold
// rule -- it almost never triggers a cascade.
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
    vector<pair<int,int>> edges(M);
    for (int i = 1; i <= N; i++) par[i] = i;
    for (int i = 0; i < M; i++){
        int u, v; scanf("%d %d", &u, &v);
        edges[i] = {u, v};
        uni(u, v);
    }
    vector<char> isRival(N + 1, 0);
    for (int i = 0; i < S; i++){ int b; scanf("%d", &b); isRival[b] = 1; }

    // group available nodes by component (arena), sorted by value desc within each
    map<int, vector<int>> comp;
    for (int v = 1; v <= N; v++) if (!isRival[v]) comp[find(v)].push_back(v);
    vector<vector<int>> arenas;
    for (auto &kv : comp){
        vector<int> nodes = kv.second;
        sort(nodes.begin(), nodes.end(), [&](int a, int b){
            if (val[a] != val[b]) return val[a] > val[b];
            return a < b;
        });
        arenas.push_back(nodes);
    }
    // order arenas by their single best available value, descending
    sort(arenas.begin(), arenas.end(), [&](const vector<int>& a, const vector<int>& b){
        return val[a[0]] > val[b[0]];
    });

    vector<int> chosen;
    int depth = 0;
    while ((int)chosen.size() < K){
        bool any = false;
        for (auto &nodes : arenas){
            if ((int)chosen.size() >= K) break;
            if (depth < (int)nodes.size()){
                chosen.push_back(nodes[depth]);
                any = true;
            }
        }
        depth++;
        if (!any) break;
    }
    for (int id : chosen) printf("%d\n", id);
    return 0;
}
