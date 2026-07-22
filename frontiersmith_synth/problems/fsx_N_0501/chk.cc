#include "testlib.h"
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
vector<int> weight, storm;
vector<Edge> edges;
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
        if (take[i]) dsu.unite(edges[i].u, edges[i].v);
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
            F += edges[i].c;
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
            const Edge& e = edges[id];
            if (e.l <= t && t <= e.r) {
                int rho = (e.a + (ll)e.b * t) % P;
                adj[e.u].push_back({e.v, rho});
                adj[e.v].push_back({e.u, rho});
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

        uint32_t bad = (1u << storm[t]) | (1u << ((storm[t] + 1) % P));
        for (int v = 2; v <= N; v++) {
            if (reach[v] == 0) {
                F += (ll)O * weight[v];
            } else if ((reach[v] & ~bad) == 0) {
                F += (ll)R * weight[v];
            }
        }
    }
    return F;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    T = inf.readInt();
    P = inf.readInt();
    O = inf.readInt();
    R = inf.readInt();

    weight.assign(N + 1, 1);
    for (int i = 1; i <= N; i++) weight[i] = inf.readInt();
    storm.assign(T, 0);
    for (int t = 0; t < T; t++) storm[t] = inf.readInt();

    edges.assign(M + 1, {});
    for (int i = 1; i <= M; i++) {
        edges[i].u = inf.readInt();
        edges[i].v = inf.readInt();
        edges[i].c = inf.readInt();
        edges[i].l = inf.readInt();
        edges[i].r = inf.readInt();
        edges[i].a = inf.readInt();
        edges[i].b = inf.readInt();
    }
    allMask = (P == 32 ? 0xffffffffu : ((1u << P) - 1u));

    vector<char> take(M + 1, 0);
    int qn = ouf.readInt(0, M, "q");
    for (int i = 0; i < qn; i++) {
        int id = ouf.readInt(1, M, "edge_index");
        if (take[id]) quitf(_wa, "edge index %d listed more than once", id);
        take[id] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    if (!connectedStatic(take)) quitf(_wa, "selected links are not statically connected");

    vector<char> base(M + 1, 0);
    for (int i = 1; i <= N - 1; i++) base[i] = 1;
    if (!connectedStatic(base)) quitf(_fail, "bad instance: fallback links are not connected");

    ll F = computeF(take);
    ll B = computeF(base);
    if (B <= 0) quitf(_fail, "bad instance: nonpositive baseline B=%lld", B);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld selected=%d Ratio: %.6f", F, B, qn, sc / 1000.0);
    return 0;
}
