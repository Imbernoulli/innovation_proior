// TIER: strong
// Balance-constrained max-cut local search with multi-restart.
//   * one restart is seeded from the greedy flow-aware placement (guarantees
//     strong >= greedy after improvement);
//   * other restarts start from random balanced assignments;
//   * improvement = alternating sweeps of feasible single FLIPS (change balance)
//     and balance-preserving SWAPS (exchange one performer per stage), both taken
//     only when they raise the cross-plaza flow and keep |T0-T1| <= tau;
//   * keep the best feasible assignment over all restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m; ll tau;
vector<int> a;
vector<vector<pair<int,int>>> adj;  // (nbr, w)
vector<ll> incid;

// maintained state during local search
vector<ll> n0, n1;   // weight of a node's neighbours currently on stage 0 / 1
ll T0, T1;

void recompute(const vector<int>& stage) {
    n0.assign(n + 1, 0); n1.assign(n + 1, 0);
    T0 = T1 = 0;
    for (int i = 1; i <= n; i++) (stage[i] == 0 ? T0 : T1) += a[i];
    for (int i = 1; i <= n; i++)
        for (auto& pr : adj[i])
            (stage[pr.first] == 0 ? n0[i] : n1[i]) += pr.second;
}

inline ll deltaFlip(int i, const vector<int>& stage) {
    // gain in cut if performer i flips to the other stage
    return (stage[i] == 0) ? (n0[i] - n1[i]) : (n1[i] - n0[i]);
}

void applyFlip(int i, vector<int>& stage) {
    int from = stage[i], to = 1 - from;
    for (auto& pr : adj[i]) {
        int j = pr.first, w = pr.second;
        if (from == 0) { n0[j] -= w; n1[j] += w; }
        else           { n1[j] -= w; n0[j] += w; }
    }
    stage[i] = to;
    if (from == 0) { T0 -= a[i]; T1 += a[i]; }
    else           { T1 -= a[i]; T0 += a[i]; }
}

ll cutOf(const vector<int>& stage) {
    ll F = 0;
    for (int i = 1; i <= n; i++)
        for (auto& pr : adj[i])
            if (i < pr.first && stage[i] != stage[pr.first]) F += pr.second;
    return F;
}

ll wBetween(int i, int j) {
    ll s = 0;
    for (auto& pr : adj[i]) if (pr.first == j) s += pr.second;
    return s;
}

mt19937 rng(20260701u);

void improve(vector<int>& stage, int maxSweeps) {
    recompute(stage);
    for (int sweep = 0; sweep < maxSweeps; sweep++) {
        bool changed = false;
        // ---- feasible single-flip sweep ----
        for (int i = 1; i <= n; i++) {
            ll g = deltaFlip(i, stage);
            if (g <= 0) continue;
            ll newDiff = (stage[i] == 0) ? llabs((T0 - a[i]) - (T1 + a[i]))
                                         : llabs((T0 + a[i]) - (T1 - a[i]));
            if (newDiff <= tau) { applyFlip(i, stage); changed = true; }
        }
        // ---- balance-preserving swap sweep ----
        vector<int> s0, s1;
        for (int i = 1; i <= n; i++) (stage[i] == 0 ? s0 : s1).push_back(i);
        if (!s0.empty() && !s1.empty()) {
            int tries = min((int)s0.size(), 400);
            for (int t = 0; t < tries; t++) {
                int i = s0[rng() % s0.size()];
                if (stage[i] != 0) continue;
                for (int rep = 0; rep < 4; rep++) {
                    int j = s1[rng() % s1.size()];
                    if (stage[j] != 1) continue;
                    ll g = deltaFlip(i, stage) + deltaFlip(j, stage) - 2 * wBetween(i, j);
                    if (g <= 0) continue;
                    ll newDiff = llabs((T0 - a[i] + a[j]) - (T1 + a[i] - a[j]));
                    if (newDiff <= tau) {
                        applyFlip(i, stage); applyFlip(j, stage);
                        changed = true; break;
                    }
                }
            }
        }
        if (!changed) break;
    }
}

// greedy flow-aware placement + feasibility repair (same idea as greedy.cpp),
// used as one strong restart seed.
vector<int> greedySeed() {
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(),
         [&](int x, int y){ return incid[x] > incid[y]; });
    vector<int> stage(n + 1, -1);
    ll t0 = 0, t1 = 0;
    for (int idx : order) {
        ll w0 = 0, w1 = 0;
        for (auto& pr : adj[idx]) {
            int s = stage[pr.first];
            if (s == 0) w0 += pr.second; else if (s == 1) w1 += pr.second;
        }
        int put;
        if (w0 > w1) put = 1; else if (w1 > w0) put = 0; else put = (t0 <= t1) ? 0 : 1;
        stage[idx] = put;
        if (put == 0) t0 += a[idx]; else t1 += a[idx];
    }
    // repair
    auto gainMove = [&](int i, int to) -> ll {
        ll d = 0;
        for (auto& pr : adj[i]) {
            int s = stage[pr.first]; if (s < 0) continue;
            bool cb = (s != stage[i]), ca = (s != to);
            if (ca && !cb) d += pr.second; if (!ca && cb) d -= pr.second;
        }
        return d;
    };
    while (llabs(t0 - t1) > tau) {
        int heavy = (t0 > t1) ? 0 : 1, light = 1 - heavy;
        int best = -1; ll bd = LLONG_MIN;
        for (int i = 1; i <= n; i++) {
            if (stage[i] != heavy) continue;
            ll d = gainMove(i, light);
            if (d > bd) { bd = d; best = i; }
        }
        if (best == -1) break;
        if (heavy == 0) { t0 -= a[best]; t1 += a[best]; } else { t1 -= a[best]; t0 += a[best]; }
        stage[best] = light;
    }
    return stage;
}

vector<int> randomBalanced() {
    vector<int> stage(n + 1, 0);
    ll t0 = 0, t1 = 0;
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    for (int i = n - 1; i > 0; i--) swap(order[i], order[rng() % (i + 1)]);
    for (int idx : order) {
        int put = (t0 <= t1) ? 0 : 1;
        // add some randomness while staying roughly balanced
        if (llabs(t0 - t1) < tau / 2 && (rng() & 1)) put = 1 - put;
        stage[idx] = put;
        if (put == 0) t0 += a[idx]; else t1 += a[idx];
    }
    // repair to feasibility
    while (llabs(t0 - t1) > tau) {
        int heavy = (t0 > t1) ? 0 : 1, light = 1 - heavy;
        int mv = -1;
        for (int i = 1; i <= n; i++) if (stage[i] == heavy) { mv = i; break; }
        if (mv == -1) break;
        stage[mv] = light;
        if (heavy == 0) { t0 -= a[mv]; t1 += a[mv]; } else { t1 -= a[mv]; t0 += a[mv]; }
    }
    return stage;
}

int main() {
    if (scanf("%d %d %lld", &n, &m, &tau) != 3) return 0;
    a.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
    adj.assign(n + 1, {});
    incid.assign(n + 1, 0);
    for (int e = 0; e < m; e++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w}); adj[v].push_back({u, w});
        incid[u] += w; incid[v] += w;
    }

    int restarts = (n <= 300) ? 10 : (n <= 800 ? 6 : 3);
    int sweeps = 30;

    vector<int> best; ll bestF = -1;

    // restart 0: greedy seed
    {
        vector<int> st = greedySeed();
        improve(st, sweeps);
        ll F = cutOf(st);
        if (llabs(T0 - T1) <= tau && F > bestF) { bestF = F; best = st; }
    }
    for (int r = 1; r < restarts; r++) {
        vector<int> st = randomBalanced();
        improve(st, sweeps);
        ll F = cutOf(st);
        if (llabs(T0 - T1) <= tau && F > bestF) { bestF = F; best = st; }
    }

    if (best.empty()) best = greedySeed();  // fallback (feasible)
    for (int i = 1; i <= n; i++) printf("%d\n", best[i]);
    return 0;
}
