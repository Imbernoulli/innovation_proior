// TIER: strong
// Build a feasible degree-bounded tree (degree-capped Kruskal, backbone fallback), then
// run edge-swap local search: for each cheap unused segment (u,v), find the most expensive
// segment on the tree path u..v and swap it out whenever the exchange keeps every fan-out
// <= D and lowers total length. Multiple passes until no improving swap remains.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D;
vector<int> eu, ev; vector<ll> ew;

int par[1600006];
int find(int x){ while(par[x]!=x){ par[x]=par[par[x]]; x=par[x];} return x; }

// tree adjacency: node -> list of (neighbor, edgeIndex)
vector<vector<pair<int,int>>> tadj;
vector<char> inTree;

int main(){
    scanf("%d %d %d", &n, &m, &D);
    eu.resize(m+1); ev.resize(m+1); ew.resize(m+1);
    for (int i = 1; i <= m; i++) scanf("%d %d %lld", &eu[i], &ev[i], &ew[i]);

    // ---- initial feasible tree ----
    vector<int> ord(m);
    iota(ord.begin(), ord.end(), 1);
    sort(ord.begin(), ord.end(), [&](int a, int b){ return ew[a] < ew[b]; });

    for (int i = 1; i <= n; i++) par[i] = i;
    vector<int> deg(n + 1, 0);
    inTree.assign(m + 1, 0);
    tadj.assign(n + 1, {});
    vector<int> chosen;
    for (int idx : ord){
        int u = eu[idx], v = ev[idx];
        if (deg[u] >= D || deg[v] >= D) continue;
        if (find(u) == find(v)) continue;
        par[find(u)] = find(v);
        deg[u]++; deg[v]++;
        chosen.push_back(idx);
        if ((int)chosen.size() == n - 1) break;
    }
    if ((int)chosen.size() != n - 1){
        // backbone fallback
        chosen.clear();
        for (int i = 0; i <= n; i++){ deg[i] = 0; }
        vector<int> bestIdx(n + 1, -1); vector<ll> bestW(n + 1, LLONG_MAX);
        for (int i = 1; i <= m; i++) if (abs(eu[i]-ev[i])==1){
            int lo = min(eu[i], ev[i]);
            if (ew[i] < bestW[lo]){ bestW[lo] = ew[i]; bestIdx[lo] = i; }
        }
        for (int i = 1; i < n; i++) chosen.push_back(bestIdx[i]);
    }
    for (int idx : chosen){
        inTree[idx] = 1;
        tadj[eu[idx]].push_back({ev[idx], idx});
        tadj[ev[idx]].push_back({eu[idx], idx});
    }

    // ---- edge-swap local search ----
    vector<int> seen(n + 1, 0), parNode(n + 1, 0), parEdge(n + 1, 0);
    int ver = 0;
    // candidate unused edges, cheapest first
    // ord already sorted ascending; reuse it.
    int passes = 6;
    for (int p = 0; p < passes; p++){
        bool improved = false;
        for (int idx : ord){
            if (inTree[idx]) continue;
            int u = eu[idx], v = ev[idx];
            ll w = ew[idx];
            // BFS u -> v on current tree
            ver++;
            seen[u] = ver; parNode[u] = 0; parEdge[u] = 0;
            static vector<int> q; q.clear(); q.push_back(u);
            int head = 0; bool found = false;
            while (head < (int)q.size()){
                int x = q[head++];
                if (x == v){ found = true; break; }
                for (auto& pr : tadj[x]){
                    int y = pr.first;
                    if (seen[y] == ver) continue;
                    seen[y] = ver; parNode[y] = x; parEdge[y] = pr.second;
                    q.push_back(y);
                }
            }
            if (!found) continue; // should not happen (tree is connected)
            // walk path v -> u, find max-weight edge
            int bestE = -1; ll bestEW = -1;
            int cur = v;
            while (cur != u){
                int pe = parEdge[cur];
                if (ew[pe] > bestEW){ bestEW = ew[pe]; bestE = pe; }
                cur = parNode[cur];
            }
            if (bestE < 0 || bestEW <= w) continue;
            int a = eu[bestE], b = ev[bestE];
            // feasibility of swap: remove bestE (a,b: -1), add idx (u,v: +1)
            int du = deg[u] + 1, dv = deg[v] + 1;
            // if u/v coincide with a/b adjust
            int da = deg[a] - 1, db = deg[b] - 1;
            // apply overlaps
            auto adj = [&](int node, int base)->int{
                int d = base;
                // this helper unused; explicit below
                return d;
            };
            (void)adj;
            // compute effective degrees for the (up to 4) distinct nodes
            // node u
            int fu = deg[u] + 1 - ((u==a)?1:0) - ((u==b)?1:0);
            int fv = deg[v] + 1 - ((v==a)?1:0) - ((v==b)?1:0);
            int fa = deg[a] - 1 + ((a==u)?1:0) + ((a==v)?1:0);
            int fb = deg[b] - 1 + ((b==u)?1:0) + ((b==v)?1:0);
            if (fu > D || fv > D || fa > D || fb > D) continue;
            (void)du;(void)dv;(void)da;(void)db;
            // perform swap: remove bestE
            auto rem = [&](int node, int e){
                auto& vv = tadj[node];
                for (size_t t = 0; t < vv.size(); t++) if (vv[t].second == e){ vv[t] = vv.back(); vv.pop_back(); break; }
            };
            rem(a, bestE); rem(b, bestE);
            inTree[bestE] = 0; deg[a]--; deg[b]--;
            inTree[idx] = 1; deg[u]++; deg[v]++;
            tadj[u].push_back({v, idx});
            tadj[v].push_back({u, idx});
            improved = true;
        }
        if (!improved) break;
    }

    // ---- emit ----
    vector<int> out;
    for (int i = 1; i <= m; i++) if (inTree[i]) out.push_back(i);
    printf("%d\n", (int)out.size());
    for (int idx : out) printf("%d\n", idx);
    return 0;
}
