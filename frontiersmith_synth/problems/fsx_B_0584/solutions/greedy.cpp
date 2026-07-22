// TIER: greedy
// The obvious approach named in the folklore: CLUSTER TASKS BY SHARED PAGES. Each task
// is bucketed by its rarest page (its cluster's private signature), and tasks are emitted
// signature-group by signature-group (a topological order that keeps each cluster together).
// This restores cluster locality but is cache-oblivious: it never looks at the LRU state,
// so a page that recurs ACROSS clusters (a globally-hot page) gets a reuse distance of a
// whole cluster and is reloaded every time. It authors the wrong reuse-distance histogram.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, C, E;
    if (scanf("%d %d %d %d", &N, &M, &C, &E) != 4) return 0;
    vector<vector<int>> pg(N + 1);
    vector<int> gcnt(M, 0);
    for (int i = 1; i <= N; i++) {
        int k; scanf("%d", &k);
        pg[i].resize(k);
        for (int j = 0; j < k; j++) { scanf("%d", &pg[i][j]); }
        for (int p : pg[i]) gcnt[p]++;
    }
    vector<vector<int>> succ(N + 1);
    vector<int> indeg(N + 1, 0);
    for (int i = 0; i < E; i++) {
        int u, v; scanf("%d %d", &u, &v);
        succ[u].push_back(v); indeg[v]++;
    }
    // primary page = rarest page of the task (smallest global frequency, tie smallest id)
    vector<long long> prio(N + 1);
    for (int i = 1; i <= N; i++) {
        int best = pg[i][0];
        for (int p : pg[i]) if (gcnt[p] < gcnt[best] || (gcnt[p] == gcnt[best] && p < best)) best = p;
        prio[i] = (long long)best;    // cluster key = signature page id
    }
    // topological order preferring (signature page, id) -> groups clusters together
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    for (int i = 1; i <= N; i++) if (indeg[i] == 0) pq.push({ prio[i], i });
    vector<int> order; order.reserve(N);
    while (!pq.empty()) {
        auto [k, id] = pq.top(); pq.pop();
        order.push_back(id);
        for (int v : succ[id]) if (--indeg[v] == 0) pq.push({ prio[v], v });
    }
    for (int i = 0; i < N; i++) printf("%d%c", order[i], i == N - 1 ? '\n' : ' ');
    return 0;
}
