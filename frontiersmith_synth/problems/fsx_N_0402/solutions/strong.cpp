// TIER: strong
// Best-improvement F-greedy (add the affordable hub that most increases lambda*bandwidth)
// followed by 1-for-1 swap local search, over a few restarts from different seed pairs.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M; ll C;
vector<ll> p, c;
vector<array<int,3>> edges;
vector<vector<pair<int,ll>>> adjPair; // not strictly needed; kept simple

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

ll costOf(const vector<int>& S) { ll s = 0; for (int x : S) s += c[x]; return s; }

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

    // rank affordable pairs; use top few as restart seeds
    vector<tuple<ll,int,int>> pairs;
    for (auto& kv : pairW) {
        int u = kv.first.first, v = kv.first.second;
        if (c[u] + c[v] <= C) pairs.push_back({kv.second * (p[u] + p[v]), u, v});
    }
    if (pairs.empty()) { printf("0\n"); return 0; }
    sort(pairs.rbegin(), pairs.rend());

    unsigned long long rngState = 0x9e3779b97f4a7c15ULL;
    auto rng = [&]() { rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17; return rngState; };

    vector<int> bestS; ll bestF = -1;
    int restarts = min((int)pairs.size(), 6);

    for (int rs = 0; rs < restarts; rs++) {
        int su = get<1>(pairs[rs]), sv = get<2>(pairs[rs]);
        vector<int> S = {su, sv};
        vector<char> in(N + 1, 0); in[su] = in[sv] = 1;
        ll cost = c[su] + c[sv];
        ll curF = evalF(S);

        // best-improvement additions
        bool improved = true;
        while (improved) {
            improved = false;
            ll bestGain = 0; int bestH = -1;
            for (int h = 1; h <= N; h++) {
                if (in[h] || cost + c[h] > C) continue;
                S.push_back(h);
                ll f = evalF(S);
                S.pop_back();
                if (f - curF > bestGain) { bestGain = f - curF; bestH = h; }
            }
            if (bestH >= 0) {
                S.push_back(bestH); in[bestH] = 1; cost += c[bestH]; curF += bestGain; improved = true;
            }
        }

        // 1-for-1 swap local search
        bool swapped = true; int guard = 0;
        while (swapped && guard++ < 40) {
            swapped = false;
            for (int si = 0; si < (int)S.size() && !swapped; si++) {
                int out = S[si];
                for (int h = 1; h <= N; h++) {
                    if (in[h]) continue;
                    if (cost - c[out] + c[h] > C) continue;
                    vector<int> T = S; T[si] = h;
                    ll f = evalF(T);
                    if (f > curF) {
                        in[out] = 0; in[h] = 1; S[si] = h;
                        cost += c[h] - c[out]; curF = f; swapped = true; break;
                    }
                }
            }
        }

        // one randomized perturbation + re-add pass
        if ((int)S.size() > 2) {
            int drop = S[rng() % S.size()];
            vector<int> T; for (int x : S) if (x != drop) T.push_back(x);
            vector<char> tin(N + 1, 0); ll tc = 0; for (int x : T) { tin[x] = 1; tc += c[x]; }
            ll tf = evalF(T);
            bool imp = true;
            while (imp) {
                imp = false; ll bg = 0; int bh = -1;
                for (int h = 1; h <= N; h++) {
                    if (tin[h] || tc + c[h] > C) continue;
                    T.push_back(h); ll f = evalF(T); T.pop_back();
                    if (f - tf > bg) { bg = f - tf; bh = h; }
                }
                if (bh >= 0) { T.push_back(bh); tin[bh] = 1; tc += c[bh]; tf += bg; imp = true; }
            }
            if (tf > curF) { S = T; curF = tf; }
        }

        if (curF > bestF) { bestF = curF; bestS = S; }
    }

    if (bestS.empty()) { printf("2\n%d %d\n", get<1>(pairs[0]), get<2>(pairs[0])); return 0; }
    printf("%d\n", (int)bestS.size());
    for (int i = 0; i < (int)bestS.size(); i++) printf("%d%c", bestS[i], i + 1 == (int)bestS.size() ? '\n' : ' ');
    return 0;
}
