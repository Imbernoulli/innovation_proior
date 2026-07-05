// TIER: strong
// Degree-discounted greedy (GWMIN, pick max w/(deg+1) on the residual graph)
// combined with local search: (0,1)-additions and (1,2)-swaps across several
// deterministically seeded randomized restarts; keep the best independent set.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> w;
vector<vector<int>> adj;

// verify + weight of a chosen set (debug-safe; not used for output correctness path)
ll weightOf(const vector<char>& chosen) {
    ll s = 0;
    for (int i = 1; i <= n; i++) if (chosen[i]) s += w[i];
    return s;
}

// GWMIN on residual graph, with a per-node tie/perturbation key from rng
vector<char> gwmin(mt19937_64& rng) {
    vector<char> removed(n + 1, 0), chosen(n + 1, 0);
    vector<int> deg(n + 1, 0);
    for (int i = 1; i <= n; i++) deg[i] = (int)adj[i].size();
    vector<double> jitter(n + 1);
    for (int i = 1; i <= n; i++)
        jitter[i] = 1.0 + 0.15 * ((double)(rng() % 1000) / 1000.0 - 0.5);

    int remaining = n;
    while (remaining > 0) {
        int best = -1; double bestKey = -1;
        for (int i = 1; i <= n; i++) {
            if (removed[i]) continue;
            double key = jitter[i] * (double)w[i] / (double)(deg[i] + 1);
            if (key > bestKey) { bestKey = key; best = i; }
        }
        if (best == -1) break;
        chosen[best] = 1;
        removed[best] = 1; remaining--;
        for (int u : adj[best]) if (!removed[u]) {
            removed[u] = 1; remaining--;
        }
        // recompute degrees on residual graph
        for (int i = 1; i <= n; i++) if (!removed[i]) {
            int d = 0;
            for (int u : adj[i]) if (!removed[u]) d++;
            deg[i] = d;
        }
    }
    return chosen;
}

// count chosen neighbors of v
int conflictCount(int v, const vector<char>& chosen) {
    int c = 0;
    for (int u : adj[v]) if (chosen[u]) c++;
    return c;
}

void localSearch(vector<char>& chosen, mt19937_64& rng) {
    bool improved = true;
    int guard = 0;
    while (improved && guard++ < 200) {
        improved = false;

        // (0,1)-addition: add any free vertex with no chosen neighbor
        for (int v = 1; v <= n; v++) {
            if (chosen[v]) continue;
            if (conflictCount(v, chosen) == 0) { chosen[v] = 1; improved = true; }
        }

        // (1,2)-swap: remove one chosen vertex x, add two non-adjacent free
        // vertices (each conflicting only with x) if total weight increases.
        vector<int> cur;
        for (int i = 1; i <= n; i++) if (chosen[i]) cur.push_back(i);
        shuffle(cur.begin(), cur.end(), rng);
        for (int x : cur) {
            if (!chosen[x]) continue;
            // candidates: free vertices whose only chosen neighbor is x
            vector<int> cand;
            for (int y : adj[x]) {
                if (chosen[y]) continue;
                if (conflictCount(y, chosen) == 1) cand.push_back(y); // only x
            }
            if ((int)cand.size() < 2) continue;
            // greedily find best non-adjacent pair by weight
            sort(cand.begin(), cand.end(), [&](int a, int b){ return w[a] > w[b]; });
            int foundA = -1, foundB = -1; ll bestGain = 0;
            for (size_t i = 0; i < cand.size(); i++) {
                for (size_t j = i + 1; j < cand.size(); j++) {
                    int a = cand[i], b = cand[j];
                    // must be non-adjacent to each other
                    bool adjab = false;
                    for (int u : adj[a]) if (u == b) { adjab = true; break; }
                    if (adjab) continue;
                    ll gain = (ll)w[a] + w[b] - w[x];
                    if (gain > bestGain) { bestGain = gain; foundA = a; foundB = b; }
                }
            }
            if (foundA != -1) {
                chosen[x] = 0; chosen[foundA] = 1; chosen[foundB] = 1;
                improved = true;
            }
        }
    }
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<char> best;
    ll bestW = -1;

    int restarts = 12;
    for (int r = 0; r < restarts; r++) {
        mt19937_64 rng(0x5a1000dULL + 1000003ULL * r + 7ULL * n + m);
        vector<char> chosen = gwmin(rng);
        localSearch(chosen, rng);
        ll ww = weightOf(chosen);
        if (ww > bestW) { bestW = ww; best = chosen; }
    }

    vector<int> sel;
    for (int i = 1; i <= n; i++) if (best[i]) sel.push_back(i);
    printf("%d\n", (int)sel.size());
    for (size_t i = 0; i < sel.size(); i++)
        printf("%d%c", sel[i], i + 1 == sel.size() ? '\n' : ' ');
    if (sel.empty()) printf("\n");
    return 0;
}
