// TIER: strong
// Start from the greedy temporal-residue build, then use exact marginal objective
// evaluations for promising links and prune redundant links while preserving
// static connectivity.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Edge {
    int u, v, c, l, r, a, b;
};

struct DSU {
    vector<int> p, sz;
    DSU(int n = 0) { init(n); }
    void init(int n) {
        p.resize(n + 1);
        sz.assign(n + 1, 1);
        iota(p.begin(), p.end(), 0);
    }
    int find(int x) {
        while (p[x] != x) {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (sz[a] < sz[b]) swap(a, b);
        p[b] = a;
        sz[a] += sz[b];
    }
};

int N, M, T, P, O, R;
vector<int> w, h;
vector<Edge> e;
uint32_t allMask;

static inline uint32_t rotMask(uint32_t x, int sh) {
    sh %= P;
    x &= allMask;
    if (sh == 0) return x;
    return ((x << sh) | (x >> (P - sh))) & allMask;
}

static bool connectedStatic(const vector<char>& take) {
    DSU dsu(N);
    for (int i = 1; i <= M; i++) {
        if (take[i]) dsu.unite(e[i].u, e[i].v);
    }
    int root = dsu.find(1);
    for (int v = 2; v <= N; v++) {
        if (dsu.find(v) != root) return false;
    }
    return true;
}

static ll computeF(const vector<char>& take) {
    ll F = 0;
    vector<int> selected;
    selected.reserve(M);
    for (int i = 1; i <= M; i++) {
        if (take[i]) {
            F += e[i].c;
            selected.push_back(i);
        }
    }

    vector<vector<pair<int, int>>> adj(N + 1);
    vector<uint32_t> reach(N + 1);
    vector<char> inq(N + 1);
    queue<int> q;
    for (int t = 0; t < T; t++) {
        for (int v = 1; v <= N; v++) {
            adj[v].clear();
            reach[v] = 0;
            inq[v] = 0;
        }
        for (int id : selected) {
            const Edge& x = e[id];
            if (x.l <= t && t <= x.r) {
                int rho = (x.a + (ll)x.b * t) % P;
                adj[x.u].push_back({x.v, rho});
                adj[x.v].push_back({x.u, rho});
            }
        }
        reach[1] = 1u;
        q.push(1);
        inq[1] = 1;
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            inq[v] = 0;
            uint32_t bits = reach[v];
            for (auto [to, rho] : adj[v]) {
                if (to == 1 && v != 1) continue;
                uint32_t nb = rotMask(bits, rho);
                if ((nb & ~reach[to]) != 0) {
                    reach[to] |= nb;
                    if (!inq[to]) {
                        inq[to] = 1;
                        q.push(to);
                    }
                }
            }
        }
        uint32_t bad = (1u << h[t]) | (1u << ((h[t] + 1) % P));
        for (int v = 2; v <= N; v++) {
            if (reach[v] == 0) F += (ll)O * w[v];
            else if ((reach[v] & ~bad) == 0) F += (ll)R * w[v];
        }
    }
    return F;
}

static ll potentialScore(int id) {
    const Edge& x = e[id];
    ll good = 0, bad = 0;
    for (int t = x.l; t <= x.r; t++) {
        int rho = (x.a + (ll)x.b * t) % P;
        bool forbidden = (rho == h[t] || rho == (h[t] + 1) % P);
        if (forbidden) bad++;
        else good++;
    }
    if (x.u == 1 || x.v == 1) {
        int other = (x.u == 1 ? x.v : x.u);
        return (good * 5 - bad * 2) * (ll)R * w[other] - 6LL * e[id].c;
    }
    return (good * 2 - bad) * (ll)R * (w[x.u] + w[x.v]) - 16LL * e[id].c;
}

int main() {
    if (!(cin >> N >> M >> T >> P)) return 0;
    cin >> O >> R;
    w.assign(N + 1, 1);
    for (int i = 1; i <= N; i++) cin >> w[i];
    h.assign(T, 0);
    for (int t = 0; t < T; t++) cin >> h[t];
    e.assign(M + 1, {});
    for (int i = 1; i <= M; i++) {
        cin >> e[i].u >> e[i].v >> e[i].c >> e[i].l >> e[i].r >> e[i].a >> e[i].b;
    }
    allMask = (1u << P) - 1u;

    vector<char> take(M + 1, 0);
    for (int i = 1; i <= N - 1; i++) take[i] = 1;

    // Same cheap recognizer as the greedy tier, used as a warm start.
    for (int id = N; id <= M; id++) {
        const Edge& x = e[id];
        ll good = 0, bad = 0;
        for (int t = x.l; t <= x.r; t++) {
            int rho = (x.a + (ll)x.b * t) % P;
            bool forbidden = (rho == h[t] || rho == (h[t] + 1) % P);
            if (forbidden) bad++;
            else good++;
        }
        bool hq = (x.u == 1 || x.v == 1);
        int other = (x.u == 1 ? x.v : x.u);
        ll potential;
        bool choose = false;
        if (hq) {
            potential = (good * 4 - bad * 2) * (ll)R * w[other];
            choose = potential > (ll)x.c * 8;
        } else {
            int mxw = max(w[x.u], w[x.v]);
            potential = (good * 2 - bad) * (ll)R * (w[x.u] + w[x.v]);
            if (mxw >= 10) choose = potential > (ll)x.c * 5;
            else choose = potential > (ll)x.c * 250;
        }
        if (choose) take[id] = 1;
    }

    ll cur = computeF(take);
    vector<pair<ll, int>> cand;
    for (int id = N; id <= M; id++) {
        ll s = potentialScore(id);
        if (s > -5000) cand.push_back({s, id});
    }
    sort(cand.begin(), cand.end(), [](const pair<ll, int>& A, const pair<ll, int>& B) {
        if (A.first != B.first) return A.first > B.first;
        return A.second < B.second;
    });

    int limit = min((int)cand.size(), 720);
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < limit; i++) {
            int id = cand[i].second;
            if (take[id]) continue;
            take[id] = 1;
            ll nf = computeF(take);
            if (nf < cur) cur = nf;
            else take[id] = 0;
        }
    }

    vector<int> chosen;
    for (int id = 1; id <= M; id++) if (take[id]) chosen.push_back(id);
    sort(chosen.begin(), chosen.end(), [&](int a, int b) {
        if (e[a].c != e[b].c) return e[a].c > e[b].c;
        return a > b;
    });

    for (int id : chosen) {
        take[id] = 0;
        if (connectedStatic(take)) {
            ll nf = computeF(take);
            if (nf < cur) cur = nf;
            else take[id] = 1;
        } else {
            take[id] = 1;
        }
    }

    vector<int> ans;
    for (int id = 1; id <= M; id++) if (take[id]) ans.push_back(id);
    cout << ans.size() << "\n";
    for (int id : ans) cout << id << "\n";
    return 0;
}
