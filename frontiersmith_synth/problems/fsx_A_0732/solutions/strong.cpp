// TIER: strong
// The insight: every arena's market nodes vote 2-vs-1 against a fixed rival
// vote, split across THREE control nodes in pairs -- a lone control seed
// always ties (1 A vs 1 B, forever); the matching PAIR of controls flips
// every market node of that pair's type in one round; all three controls
// flip the whole arena. So: identify the 3 controls and the rival node per
// arena (a control has NO rival neighbour, a market node has exactly one),
// tally each of the 3 control-pairs' total backed value, and choose per
// arena between "back the single best pair" (cost 2) or "back all three"
// (cost 3) or skip -- a 0/1 GROUP KNAPSACK over the shared budget, not a
// per-arena-independent greedy. Any leftover budget buys the best remaining
// individual market node.
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
    vector<vector<int>> adj(N + 1);
    for (int i = 1; i <= N; i++) par[i] = i;
    for (int i = 0; i < M; i++){
        int u, v; scanf("%d %d", &u, &v);
        edges[i] = {u, v};
        adj[u].push_back(v); adj[v].push_back(u);
        uni(u, v);
    }
    vector<char> isRival(N + 1, 0);
    for (int i = 0; i < S; i++){ int b; scanf("%d", &b); isRival[b] = 1; }

    // classify every available node: prize (exactly 1 rival neighbour) or control (0)
    vector<int> rivalNeighCount(N + 1, 0);
    for (int v = 1; v <= N; v++)
        for (int u : adj[v]) if (isRival[u]) rivalNeighCount[v]++;

    map<int, vector<int>> controlsOf, prizeOf;
    for (int v = 1; v <= N; v++){
        if (isRival[v]) continue;
        int r = find(v);
        if (rivalNeighCount[v] == 1) prizeOf[r].push_back(v);
        else controlsOf[r].push_back(v);
    }

    struct Option { ll val; int cost; vector<int> nodes; };
    struct Arena { vector<Option> options; };   // options[0]=partial best pair, [1]=full
    vector<Arena> arenas;

    for (auto &kv : controlsOf){
        int r = kv.first;
        vector<int> &ctl = kv.second;
        if ((int)ctl.size() != 3) continue;      // malformed / defensive skip
        vector<int> &pr = prizeOf[r];
        map<int, int> ctlIdx;
        for (int i = 0; i < 3; i++) ctlIdx[ctl[i]] = i;
        ll pairVal[3] = {0, 0, 0};  // 0:{0,1} 1:{0,2} 2:{1,2}
        for (int p : pr){
            int a = -1, b = -1;
            for (int u : adj[p]){
                auto it = ctlIdx.find(u);
                if (it != ctlIdx.end()){ if (a == -1) a = it->second; else b = it->second; }
            }
            if (a == -1 || b == -1) continue;   // defensive
            if (a > b) swap(a, b);
            int type = (a == 0 && b == 1) ? 0 : (a == 0 && b == 2) ? 1 : 2;
            pairVal[type] += val[p];
        }
        int best = 0;
        for (int t = 1; t < 3; t++) if (pairVal[t] > pairVal[best]) best = t;
        int ia = (best == 0) ? 0 : (best == 1) ? 0 : 1;
        int ib = (best == 0) ? 1 : (best == 1) ? 2 : 2;
        Arena ar;
        ar.options.push_back({pairVal[best], 2, {ctl[ia], ctl[ib]}});
        ll full = pairVal[0] + pairVal[1] + pairVal[2];
        ar.options.push_back({full, 3, {ctl[0], ctl[1], ctl[2]}});
        arenas.push_back(ar);
    }

    int G = (int)arenas.size();
    // dp[g][b] = best value using first g arenas with budget b; choice[g][b] in {-1(skip),0(partial),1(full)}
    vector<vector<ll>> dp(G + 1, vector<ll>(K + 1, 0));
    vector<vector<signed char>> choice(G + 1, vector<signed char>(K + 1, -1));
    for (int g = 1; g <= G; g++){
        for (int b = 0; b <= K; b++){
            ll bestv = dp[g - 1][b]; signed char bestc = -1;
            for (int oi = 0; oi < (int)arenas[g - 1].options.size(); oi++){
                Option &op = arenas[g - 1].options[oi];
                if (op.cost <= b){
                    ll cand = dp[g - 1][b - op.cost] + op.val;
                    if (cand > bestv){ bestv = cand; bestc = (signed char)oi; }
                }
            }
            dp[g][b] = bestv; choice[g][b] = bestc;
        }
    }
    vector<int> chosen;
    vector<char> used(N + 1, 0);
    int b = K;
    for (int g = G; g >= 1; g--){
        signed char c = choice[g][b];
        if (c >= 0){
            Option &op = arenas[g - 1].options[c];
            for (int id : op.nodes) { chosen.push_back(id); used[id] = 1; }
            b -= op.cost;
        }
    }
    int budgetLeft = K - (int)chosen.size();
    if (budgetLeft > 0){
        vector<int> rest;
        for (int v = 1; v <= N; v++) if (!isRival[v] && !used[v]) rest.push_back(v);
        sort(rest.begin(), rest.end(), [&](int a2, int b2){
            if (val[a2] != val[b2]) return val[a2] > val[b2];
            return a2 < b2;
        });
        for (int i = 0; i < budgetLeft && i < (int)rest.size(); i++) chosen.push_back(rest[i]);
    }
    for (int id : chosen) printf("%d\n", id);
    return 0;
}
