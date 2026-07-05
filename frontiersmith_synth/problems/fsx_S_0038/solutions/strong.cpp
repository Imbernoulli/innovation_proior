// TIER: strong
// Cost-effective greedy with randomized-restart local search:
//   * each restart runs a shuffled cost-per-new-cell greedy,
//   * then prunes redundant towers (all their cells doubly covered),
//   * keeping the cheapest feasible cover across restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, M, r;
    if (scanf("%d %d %d", &N, &M, &r) != 3) return 0;
    vector<ll> c(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &c[i]);
    vector<vector<int>> adj(N + 1);
    for (int e = 0; e < M; e++) { int u, v; scanf("%d %d", &u, &v); adj[u].push_back(v); adj[v].push_back(u); }

    // coverage set of each candidate site (cells within radius r)
    vector<vector<int>> cover(N + 1);
    vector<int> dist(N + 1, -1);
    for (int s = 1; s <= N; s++) {
        vector<int> touched;
        dist[s] = 0; queue<int> q; q.push(s); touched.push_back(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            cover[s].push_back(u);
            if (dist[u] == r) continue;
            for (int v : adj[u]) if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); touched.push_back(v); }
        }
        for (int u : touched) dist[u] = -1;
    }

    std::mt19937 rng(1234567u);
    ll bestCost = LLONG_MAX;
    vector<int> bestSel;

    vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i + 1;

    int restarts = (N <= 120) ? 40 : 15;
    for (int it = 0; it < restarts; it++) {
        if (it > 0) std::shuffle(perm.begin(), perm.end(), rng);

        vector<char> covered(N + 1, 0);
        vector<char> insel(N + 1, 0);
        int need = N;
        vector<int> sel;

        // cost-effective greedy: minimize cost / newly-covered
        while (need > 0) {
            int best = -1; ll bnum = 1, bden = 0; // ratio = bnum/bden = cost/new
            for (int t : perm) {
                if (insel[t]) continue;
                int newc = 0;
                for (int u : cover[t]) if (!covered[u]) newc++;
                if (newc == 0) continue;
                ll num = c[t], den = newc;
                bool better;
                if (best == -1) better = true;
                else better = (num * bden < bnum * den); // strict -> shuffled ties diverge
                if (better) { best = t; bnum = num; bden = den; }
            }
            insel[best] = 1; sel.push_back(best);
            for (int u : cover[best]) if (!covered[u]) { covered[u] = 1; need--; }
        }

        // redundancy pruning: drop a tower if all its cells are covered >=2 times
        vector<int> cnt(N + 1, 0);
        for (int t : sel) for (int u : cover[t]) cnt[u]++;
        // try to remove the most expensive towers first
        sort(sel.begin(), sel.end(), [&](int a, int b) { return c[a] > c[b]; });
        vector<char> removed(N + 1, 0);
        for (int t : sel) {
            bool canRemove = true;
            for (int u : cover[t]) if (cnt[u] < 2) { canRemove = false; break; }
            if (canRemove) { removed[t] = 1; for (int u : cover[t]) cnt[u]--; }
        }

        ll cost = 0;
        vector<int> finalSel;
        for (int t : sel) if (!removed[t]) { finalSel.push_back(t); cost += c[t]; }

        if (cost < bestCost) { bestCost = cost; bestSel = finalSel; }
    }

    printf("%d\n", (int)bestSel.size());
    for (int x : bestSel) printf("%d\n", x);
    return 0;
}
