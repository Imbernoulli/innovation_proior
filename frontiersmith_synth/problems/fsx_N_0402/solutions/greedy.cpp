// TIER: greedy
// Coverage-priority greedy: start from the best affordable pair, then add affordable
// hubs in decreasing bandwidth, keeping only additions that do not decrease F.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M; ll C;
vector<ll> p, c;
vector<array<int,3>> edges;

ll stoerWagner(int n, vector<vector<ll>> g) {
    if (n < 2) return 0;
    ll best = LLONG_MAX;
    vector<int> vert(n); for (int i = 0; i < n; i++) vert[i] = i;
    int cur = n;
    while (cur > 1) {
        vector<ll> w(cur, 0); vector<char> in(cur, 0);
        int prev = -1, last = -1;
        for (int it = 0; it < cur; it++) {
            int sel = -1;
            for (int j = 0; j < cur; j++) if (!in[j] && (sel < 0 || w[j] > w[sel])) sel = j;
            in[sel] = 1; prev = last; last = sel;
            if (it == cur - 1) best = min(best, w[sel]);
            else for (int j = 0; j < cur; j++) if (!in[j]) w[j] += g[vert[sel]][vert[j]];
        }
        for (int j = 0; j < cur; j++) {
            g[vert[prev]][vert[j]] += g[vert[last]][vert[j]];
            g[vert[j]][vert[prev]] += g[vert[j]][vert[last]];
        }
        vert.erase(vert.begin() + last); cur--;
    }
    return best == LLONG_MAX ? 0 : best;
}

ll evalF(const vector<int>& S) {
    if ((int)S.size() < 2) return 0;
    int n = S.size();
    vector<int> loc(N + 1, -1);
    for (int i = 0; i < n; i++) loc[S[i]] = i;
    vector<vector<ll>> g(n, vector<ll>(n, 0));
    ll cover = 0;
    for (int x : S) cover += p[x];
    for (auto& e : edges) {
        int u = e[0], v = e[1];
        if (loc[u] >= 0 && loc[v] >= 0) { g[loc[u]][loc[v]] += e[2]; g[loc[v]][loc[u]] += e[2]; }
    }
    return stoerWagner(n, g) * cover;
}

int main() {
    if (scanf("%d %d %lld", &N, &M, &C) != 3) return 0;
    p.assign(N + 1, 0); c.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%lld %lld", &p[i], &c[i]);
    edges.resize(M);
    map<pair<int,int>, ll> pairW;
    for (int j = 0; j < M; j++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        edges[j] = {u, v, (int)w};
        pairW[{min(u,v), max(u,v)}] += w;
    }
    ll best = -1; int bu = -1, bv = -1;
    for (auto& kv : pairW) {
        int u = kv.first.first, v = kv.first.second;
        if (c[u] + c[v] <= C) { ll cand = kv.second * (p[u] + p[v]); if (cand > best) { best = cand; bu = u; bv = v; } }
    }
    if (bu < 0) { printf("0\n"); return 0; }

    vector<int> S = {bu, bv};
    vector<char> in(N + 1, 0); in[bu] = in[bv] = 1;
    ll cost = c[bu] + c[bv];
    ll curF = evalF(S);

    // candidates by decreasing bandwidth
    vector<int> order;
    for (int i = 1; i <= N; i++) if (!in[i]) order.push_back(i);
    sort(order.begin(), order.end(), [&](int a, int b){ return p[a] > p[b]; });

    for (int h : order) {
        if (cost + c[h] > C) continue;
        S.push_back(h); in[h] = 1;
        ll f = evalF(S);
        if (f >= curF) { curF = f; cost += c[h]; }
        else { S.pop_back(); in[h] = 0; }
    }

    printf("%d\n", (int)S.size());
    for (int i = 0; i < (int)S.size(); i++) printf("%d%c", S[i], i + 1 == (int)S.size() ? '\n' : ' ');
    return 0;
}
