// TIER: strong
// GWMIN-style weight/degree greedy with randomized multi-restart local search.
// Each restart builds an independent set by repeatedly taking the alive route that
// maximizes profit/(alive-degree+1) (with perturbed tie-breaking), then improves it
// with free-add passes and (1 -> many) swap moves. The best set over all restarts wins.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> w;
vector<vector<int>> g;

mt19937 rng(0xC0FFEEu);

// greedily pick a high-weight independent subset among `cand` (which are pairwise
// checked here); returns chosen list and total weight.
static ll greedyIndepSubset(vector<int>& cand, vector<int>& out) {
    sort(cand.begin(), cand.end(), [](int a, int b){ return w[a] > w[b]; });
    static vector<char> mark;
    if ((int)mark.size() < n + 1) mark.assign(n + 1, 0);
    vector<int> touched;
    ll tot = 0;
    for (int v : cand) {
        if (mark[v]) continue;              // adjacent to something already picked
        out.push_back(v); tot += w[v];
        mark[v] = 1; touched.push_back(v);
        for (int u : g[v]) if (!mark[u]) { mark[u] = 1; touched.push_back(u); }
    }
    for (int v : touched) mark[v] = 0;
    return tot;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        g[u].push_back(v); g[v].push_back(u);
    }

    vector<int> fullDeg(n + 1);
    for (int i = 1; i <= n; i++) fullDeg[i] = (int)g[i].size();

    vector<char> selBest;                    // best selection so far
    ll bestTot = -1;

    vector<char> sel(n + 1, 0);
    vector<int>  selCnt(n + 1, 0);           // # of selected neighbors
    vector<char> alive(n + 1, 0);
    vector<int>  deg(n + 1, 0);

    auto totalOf = [&](const vector<char>& s){ ll t=0; for(int i=1;i<=n;i++) if(s[i]) t+=w[i]; return t; };

    int restarts = 24;
    for (int rs = 0; rs < restarts; rs++) {
        // ---- GWMIN construction ----
        for (int i = 1; i <= n; i++) { alive[i] = 1; deg[i] = fullDeg[i]; sel[i] = 0; }
        double jitter = (rs == 0) ? 0.0 : 0.35;   // first restart deterministic, rest perturbed

        int aliveCnt = n;
        while (aliveCnt > 0) {
            int pick = -1; double bestKey = -1;
            for (int v = 1; v <= n; v++) {
                if (!alive[v]) continue;
                double factor = 1.0;
                if (jitter > 0) factor = 1.0 + jitter * ((double)rng() / rng.max() - 0.5) * 2.0;
                double key = (double)w[v] * factor / (double)(deg[v] + 1);
                if (key > bestKey) { bestKey = key; pick = v; }
            }
            if (pick < 0) break;
            sel[pick] = 1;
            // kill pick and its alive neighbors
            vector<int> toKill; toKill.push_back(pick);
            for (int u : g[pick]) if (alive[u]) toKill.push_back(u);
            for (int x : toKill) {
                if (!alive[x]) continue;
                alive[x] = 0; aliveCnt--;
                for (int y : g[x]) if (alive[y]) deg[y]--;
            }
        }

        // ---- local search ----
        for (int i = 1; i <= n; i++) selCnt[i] = 0;
        for (int v = 1; v <= n; v++) if (sel[v]) for (int u : g[v]) selCnt[u]++;

        bool changed = true;
        int guard = 0;
        while (changed && guard++ < 40) {
            changed = false;
            // free-add: any vertex with no selected neighbor can be added
            for (int v = 1; v <= n; v++) {
                if (!sel[v] && selCnt[v] == 0) {
                    sel[v] = 1;
                    for (int u : g[v]) selCnt[u]++;
                    changed = true;
                }
            }
            // (1 -> many) swaps: drop a selected u, add the best independent subset
            // of the neighbors that were blocked only by u.
            for (int u = 1; u <= n; u++) {
                if (!sel[u]) continue;
                vector<int> cand;
                for (int x : g[u])
                    if (!sel[x] && selCnt[x] == 1) cand.push_back(x); // blocked only by u
                if (cand.empty()) continue;
                vector<int> chosen;
                ll gain = greedyIndepSubset(cand, chosen);
                if (gain > w[u]) {
                    // remove u
                    sel[u] = 0;
                    for (int y : g[u]) selCnt[y]--;
                    // add chosen
                    for (int c : chosen) {
                        sel[c] = 1;
                        for (int y : g[c]) selCnt[y]++;
                    }
                    changed = true;
                }
            }
        }

        ll tot = totalOf(sel);
        if (tot > bestTot) { bestTot = tot; selBest = sel; }
    }

    int c = 0;
    for (int i = 1; i <= n; i++) if (selBest[i]) c++;
    printf("%d\n", c);
    for (int i = 1; i <= n; i++) if (selBest[i]) printf("%d\n", i);
    return 0;
}
