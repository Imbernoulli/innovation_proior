// TIER: strong
// Cost-aware greedy (maximize new-coverage-per-credit) followed by redundancy
// pruning (drop the most expensive still-redundant relays) and a swap pass that
// replaces a relay with a cheaper one covering its unique systems. Spends fewer
// credits than the coverage-only greedy, and diverges from it per-test.
#include <bits/stdc++.h>
using namespace std;

int N, M, r;
vector<vector<int>> g;
vector<long> cost;

vector<int> ball(int src) {
    vector<int> dist(N + 1, -1);
    deque<int> q; dist[src] = 0; q.push_back(src);
    vector<int> res;
    while (!q.empty()) {
        int u = q.front(); q.pop_front();
        res.push_back(u);
        if (dist[u] >= r) continue;
        for (int v : g[u]) if (dist[v] < 0) { dist[v] = dist[u] + 1; q.push_back(v); }
    }
    return res;
}

int main() {
    scanf("%d %d %d", &N, &M, &r);
    g.assign(N + 1, {});
    for (int i = 0; i < M; i++) { int u, v; scanf("%d %d", &u, &v); g[u].push_back(v); g[v].push_back(u); }
    cost.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%d", &cost[i]);

    vector<vector<int>> B(N + 1);
    for (int i = 1; i <= N; i++) B[i] = ball(i);

    // ---- cost-ratio greedy ----
    vector<char> covered(N + 1, 0);
    int remaining = N;
    vector<char> chosen(N + 1, 0);
    while (remaining > 0) {
        int best = -1; double bestScore = -1;
        for (int i = 1; i <= N; i++) {
            if (chosen[i]) continue;
            int nw = 0;
            for (int u : B[i]) if (!covered[u]) nw++;
            if (nw == 0) continue;
            double sc = (double)nw / (double)cost[i];  // coverage per credit
            if (sc > bestScore || (sc == bestScore && best != -1 && cost[i] < cost[best])) {
                bestScore = sc; best = i;
            }
        }
        chosen[best] = 1;
        for (int u : B[best]) if (!covered[u]) { covered[u] = 1; remaining--; }
    }

    // coverCount[node] = number of chosen relays whose ball contains node
    vector<int> coverCount(N + 1, 0);
    vector<int> sel;
    for (int i = 1; i <= N; i++) if (chosen[i]) { sel.push_back(i); for (int u : B[i]) coverCount[u]++; }

    // ---- redundancy pruning: drop most expensive redundant relay first ----
    bool changed = true;
    while (changed) {
        changed = false;
        // consider relays in decreasing cost order
        sort(sel.begin(), sel.end(), [&](int a, int b){ return cost[a] > cost[b]; });
        for (int idx = 0; idx < (int)sel.size(); idx++) {
            int i = sel[idx];
            bool redundant = true;
            for (int u : B[i]) if (coverCount[u] <= 1) { redundant = false; break; }
            if (redundant) {
                for (int u : B[i]) coverCount[u]--;
                chosen[i] = 0;
                sel.erase(sel.begin() + idx);
                changed = true;
                break;
            }
        }
    }

    // ---- swap pass: for each relay, try replacing it with a cheaper single relay
    //      that still covers the systems only it covers ----
    changed = true;
    int guard = 0;
    while (changed && guard++ < 4) {
        changed = false;
        for (int i : vector<int>(sel)) {
            // unique systems covered only by i
            vector<int> uniq;
            for (int u : B[i]) if (coverCount[u] == 1) uniq.push_back(u);
            // find cheapest node j != i whose ball covers all uniq
            int bestJ = -1; long bestC = cost[i];
            for (int j = 1; j <= N; j++) {
                if (chosen[j]) continue;
                if (cost[j] >= bestC) continue;
                // does ball(j) cover every uniq node?
                bool ok = true;
                // mark ball j
                static vector<char> mark;
                mark.assign(N + 1, 0);
                for (int u : B[j]) mark[u] = 1;
                for (int u : uniq) if (!mark[u]) { ok = false; break; }
                if (ok) { bestC = cost[j]; bestJ = j; }
            }
            if (bestJ != -1) {
                // swap i -> bestJ
                for (int u : B[i]) coverCount[u]--;
                chosen[i] = 0;
                chosen[bestJ] = 1;
                for (int u : B[bestJ]) coverCount[u]++;
                sel.erase(find(sel.begin(), sel.end(), i));
                sel.push_back(bestJ);
                changed = true;
                break;
            }
        }
    }

    printf("%d\n", (int)sel.size());
    for (int x : sel) printf("%d\n", x);
    return 0;
}
