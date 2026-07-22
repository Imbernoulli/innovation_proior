// TIER: greedy
// One-pass marginal greedy: score obvious clean relays, then add a candidate only
// if an exact objective evaluation says it improves the fallback tree.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Edge {
    int u, v, c, l, r, a, b;
};

int main() {
    int N, M, T, P;
    if (!(cin >> N >> M >> T >> P)) return 0;
    int O, R;
    cin >> O >> R;
    vector<int> w(N + 1);
    for (int i = 1; i <= N; i++) cin >> w[i];
    vector<int> h(T);
    for (int t = 0; t < T; t++) cin >> h[t];
    vector<Edge> e(M + 1);
    for (int i = 1; i <= M; i++) {
        cin >> e[i].u >> e[i].v >> e[i].c >> e[i].l >> e[i].r >> e[i].a >> e[i].b;
    }

    uint32_t allMask = (1u << P) - 1u;
    auto rotMask = [&](uint32_t x, int sh) {
        sh %= P;
        x &= allMask;
        if (sh == 0) return x;
        return ((x << sh) | (x >> (P - sh))) & allMask;
    };
    auto computeF = [&](const vector<char>& take) {
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
            uint32_t badMask = (1u << h[t]) | (1u << ((h[t] + 1) % P));
            for (int v = 2; v <= N; v++) {
                if (reach[v] == 0) F += (ll)O * w[v];
                else if ((reach[v] & ~badMask) == 0) F += (ll)R * w[v];
            }
        }
        return F;
    };
    auto potentialScore = [&](int id) {
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
            return (good * 5 - bad * 2) * (ll)R * w[other] - 6LL * x.c;
        }
        int mxw = max(w[x.u], w[x.v]);
        ll mult = (mxw >= 10 ? 3 : 1);
        return mult * (good * 2 - bad) * (ll)R * (w[x.u] + w[x.v]) - 18LL * x.c;
    };

    vector<char> take(M + 1, 0);
    for (int i = 1; i <= N - 1; i++) take[i] = 1;
    ll cur = computeF(take);

    vector<pair<ll, int>> cand;
    for (int id = N; id <= M; id++) {
        ll s = potentialScore(id);
        if (s > 0) cand.push_back({s, id});
    }
    sort(cand.begin(), cand.end(), [](const pair<ll, int>& A, const pair<ll, int>& B) {
        if (A.first != B.first) return A.first > B.first;
        return A.second < B.second;
    });

    int limit = min((int)cand.size(), 8);
    for (int i = 0; i < limit; i++) {
        int id = cand[i].second;
        take[id] = 1;
        ll nf = computeF(take);
        if (nf < cur) cur = nf;
        else take[id] = 0;
    }

    vector<int> ans;
    for (int i = 1; i <= M; i++) if (take[i]) ans.push_back(i);
    cout << ans.size() << "\n";
    for (int id : ans) cout << id << "\n";
    return 0;
}
