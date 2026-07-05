// TIER: strong
// Cost-ratio greedy (newly-covered per unit build cost) with a lazy max-heap, followed by
// redundancy pruning: drop, most-expensive first, any tower whose entire ball is still
// covered by at least one other selected tower. Cost-awareness + pruning beats coverage-
// greedy, especially on cost-skewed instances, and produces a different per-test profile.
#include <bits/stdc++.h>
using namespace std;

int N, M, R;
vector<int> cost;
vector<vector<int>> adj;

int stampVal = 0;
vector<int> stamp;
void ball(int src, vector<int>& out) {
    out.clear();
    stampVal++;
    static vector<int> fr, nx;
    fr.clear();
    stamp[src] = stampVal; out.push_back(src); fr.push_back(src);
    for (int d = 0; d < R; d++) {
        nx.clear();
        for (int u : fr)
            for (int w : adj[u])
                if (stamp[w] != stampVal) {
                    stamp[w] = stampVal;
                    out.push_back(w);
                    nx.push_back(w);
                }
        fr.swap(nx);
    }
}

int main() {
    scanf("%d %d %d", &N, &M, &R);
    cost.assign(N + 1, 0);
    for (int v = 1; v <= N; v++) scanf("%d", &cost[v]);
    adj.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int a, b; scanf("%d %d", &a, &b);
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    stamp.assign(N + 1, 0);

    vector<int> gain(N + 1, 0);
    vector<int> bv;
    for (int v = 1; v <= N; v++) { ball(v, bv); gain[v] = (int)bv.size(); }

    // key = gain / cost  (higher is better)
    auto key = [&](int v) { return (double)gain[v] / (double)cost[v]; };

    vector<char> covered(N + 1, 0), chosen(N + 1, 0);
    priority_queue<pair<double,int>> pq;
    for (int v = 1; v <= N; v++) pq.push({key(v), v});

    int uncovered = N;
    vector<int> result;
    vector<int> bu;
    while (uncovered > 0 && !pq.empty()) {
        auto [kk, v] = pq.top(); pq.pop();
        if (chosen[v]) continue;
        double cur = key(v);
        if (kk != cur) { pq.push({cur, v}); continue; }
        chosen[v] = 1;
        result.push_back(v);
        ball(v, bv);
        for (int u : bv) {
            if (!covered[u]) {
                covered[u] = 1;
                uncovered--;
                ball(u, bu);
                for (int w : bu) gain[w]--;
            }
        }
    }

    // ---- redundancy pruning ----
    // covCount[u] = number of selected towers currently covering u.
    vector<int> covCount(N + 1, 0);
    for (int t : result) { ball(t, bv); for (int u : bv) covCount[u]++; }

    // most-expensive first: dropping a costly redundant tower saves the most.
    sort(result.begin(), result.end(), [&](int a, int b) { return cost[a] > cost[b]; });
    vector<char> kept(N + 1, 1);
    for (int t : result) {
        ball(t, bv);
        bool redundant = true;
        for (int u : bv) if (covCount[u] < 2) { redundant = false; break; }
        if (redundant) {
            kept[t] = 0;
            for (int u : bv) covCount[u]--;
        }
    }

    vector<int> finalSet;
    for (int t : result) if (kept[t]) finalSet.push_back(t);

    printf("%d\n", (int)finalSet.size());
    for (size_t i = 0; i < finalSet.size(); i++)
        printf("%d%c", finalSet[i], i + 1 == finalSet.size() ? '\n' : ' ');
    if (finalSet.empty()) printf("\n");
    return 0;
}
