// TIER: strong
// Degree-constrained nearest-neighbour path as a seed, then tree edge-swap local search.
// Each swap adds a cheap non-tree link {u,v} (both with a free contact slot -> exploits the
// degree-3 branching that a pure path cannot) and drops the most expensive link on the tree
// path between u and v. Every swap preserves a feasible spanning tree and never increases cost.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n;
vector<ll> X, Y;
vector<int> B;

static inline ll d2(int a, int b) {
    ll dx = X[a]-X[b], dy = Y[a]-Y[b];
    return dx*dx + dy*dy;
}

int main() {
    if (scanf("%d", &n) != 1) return 0;
    X.resize(n); Y.resize(n); B.resize(n);
    for (int i = 0; i < n; i++) scanf("%lld %lld %d", &X[i], &Y[i], &B[i]);

    // ---- seed: nearest-neighbour Hamiltonian path ----
    vector<char> used(n, 0);
    vector<int> path; path.reserve(n);
    int cur = 0; used[0] = 1; path.push_back(0);
    for (int s = 1; s < n; s++) {
        int best = -1; ll bd = LLONG_MAX;
        for (int j = 0; j < n; j++) if (!used[j]) { ll dd = d2(cur,j); if (dd < bd){bd=dd;best=j;} }
        used[best] = 1; path.push_back(best); cur = best;
    }

    vector<vector<int>> adj(n);
    vector<int> deg(n, 0);
    for (int i = 1; i < n; i++) {
        int a = path[i-1], b = path[i];
        adj[a].push_back(b); adj[b].push_back(a);
        deg[a]++; deg[b]++;
    }

    // ---- k nearest neighbours per node ----
    const int K = 6;
    vector<vector<int>> knn(n);
    {
        vector<int> idx(n);
        for (int u = 0; u < n; u++) {
            int cnt = 0;
            for (int j = 0; j < n; j++) if (j != u) idx[cnt++] = j;
            int kk = min(K, cnt);
            partial_sort(idx.begin(), idx.begin()+kk, idx.begin()+cnt,
                         [&](int a, int b){ return d2(u,a) < d2(u,b); });
            knn[u].assign(idx.begin(), idx.begin()+kk);
        }
    }

    // ---- edge-swap local search ----
    vector<int> stamp(n, 0), par(n, -1);
    int ver = 0;
    auto findPathMaxEdge = [&](int u, int v, int &pp, int &qq) -> ll {
        // BFS from u until v reached; returns max d2 edge on u..v path, endpoints in pp,qq.
        ver++;
        stamp[u] = ver; par[u] = -1;
        vector<int> queue; queue.reserve(64);
        queue.push_back(u);
        size_t head = 0;
        bool found = false;
        while (head < queue.size()) {
            int x = queue[head++];
            if (x == v) { found = true; break; }
            for (int y : adj[x]) if (stamp[y] != ver) {
                stamp[y] = ver; par[y] = x; queue.push_back(y);
            }
        }
        if (!found) return -1;
        ll mx = -1; pp = qq = -1;
        int c = v;
        while (par[c] != -1) {
            int p = par[c];
            ll w = d2(c, p);
            if (w > mx) { mx = w; pp = c; qq = p; }
            c = p;
        }
        return mx;
    };
    auto hasEdge = [&](int u, int v) -> bool {
        for (int y : adj[u]) if (y == v) return true;
        return false;
    };
    auto removeEdge = [&](int a, int b) {
        auto &A = adj[a]; A.erase(find(A.begin(), A.end(), b));
        auto &Bv = adj[b]; Bv.erase(find(Bv.begin(), Bv.end(), a));
        deg[a]--; deg[b]--;
    };

    for (int pass = 0; pass < 6; pass++) {
        bool improved = false;
        for (int u = 0; u < n; u++) {
            for (int v : knn[u]) {
                if (u == v) continue;
                if (deg[u] >= B[u] || deg[v] >= B[v]) continue;
                if (hasEdge(u, v)) continue;
                ll cuv = d2(u, v);
                int p, q;
                ll mx = findPathMaxEdge(u, v, p, q);
                if (mx > cuv && p != -1) {
                    removeEdge(p, q);
                    adj[u].push_back(v); adj[v].push_back(u);
                    deg[u]++; deg[v]++;
                    improved = true;
                }
            }
        }
        if (!improved) break;
    }

    // ---- emit ----
    vector<pair<int,int>> edges;
    for (int u = 0; u < n; u++)
        for (int v : adj[u]) if (u < v) edges.push_back({u, v});

    printf("%d\n", (int)edges.size());
    for (auto &e : edges) printf("%d %d\n", e.first + 1, e.second + 1);
    return 0;
}
