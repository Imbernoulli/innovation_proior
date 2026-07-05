// TIER: greedy
// Coverage-greedy (cost-agnostic): repeatedly build a tower on the cell that newly covers
// the most currently-uncovered cells, using a lazy max-heap keyed by coverage gain.
#include <bits/stdc++.h>
using namespace std;

int N, M, R;
vector<int> cost;
vector<vector<int>> adj;

int stampVal = 0;
vector<int> stamp;
// Return ball(src) = cells within R hops of src (including src).
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

    vector<char> covered(N + 1, 0), chosen(N + 1, 0);
    priority_queue<pair<int,int>> pq; // (gain, vertex)
    for (int v = 1; v <= N; v++) pq.push({gain[v], v});

    int uncovered = N;
    vector<int> result;
    vector<int> bu;
    while (uncovered > 0 && !pq.empty()) {
        auto [g, v] = pq.top(); pq.pop();
        if (chosen[v]) continue;
        if (g != gain[v]) { pq.push({gain[v], v}); continue; }
        // select v
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

    printf("%d\n", (int)result.size());
    for (size_t i = 0; i < result.size(); i++)
        printf("%d%c", result[i], i + 1 == result.size() ? '\n' : ' ');
    if (result.empty()) printf("\n");
    return 0;
}
