// TIER: greedy
// The obvious approach: build a MINIMUM SPANNING TREE over the static
// geometric floor w(u,v) only (Prim's algorithm from the reference element),
// completely ignoring drift rates and timing. Then serialize it the natural
// way a programmer who has not thought about the schedule would: level order
// (BFS by tree depth, ties broken by node index). Never re-touches anything.
// This is exactly the "static-MST abstraction" the family's innovation hook
// says is insufficient: it can leave a fast-drifting hub idle for many steps
// between two of its own uses whenever other branches get processed first in
// the same BFS level, paying a large stale-clock surcharge it never sees.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N; ll T;
    scanf("%d %lld", &N, &T);
    vector<ll> X(N), Y(N), D(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld %lld", &X[i], &Y[i], &D[i]);

    auto w = [&](int a, int b) { return llabs(X[a] - X[b]) + llabs(Y[a] - Y[b]); };

    // ---- Prim's MST from node 0, static weight only ----
    vector<ll> minEdge(N, LLONG_MAX);
    vector<int> parent(N, -1);
    vector<char> inTree(N, 0);
    minEdge[0] = 0;
    for (int it = 0; it < N; it++) {
        int u = -1;
        for (int i = 0; i < N; i++)
            if (!inTree[i] && (u == -1 || minEdge[i] < minEdge[u])) u = i;
        inTree[u] = 1;
        for (int v = 0; v < N; v++) {
            if (!inTree[v]) {
                ll ww = w(u, v);
                if (ww < minEdge[v]) { minEdge[v] = ww; parent[v] = u; }
            }
        }
    }

    // ---- depth of every node via parent chain ----
    vector<int> depth(N, 0);
    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i;
    // compute depth by repeated relaxation (tree, so N passes is safe/simple)
    vector<char> known(N, 0); known[0] = 1; depth[0] = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 1; i < N; i++) {
            if (!known[i] && known[parent[i]]) { depth[i] = depth[parent[i]] + 1; known[i] = 1; changed = true; }
        }
    }

    // level order: sort non-root nodes by (depth, index) ascending
    vector<int> nodes;
    for (int i = 1; i < N; i++) nodes.push_back(i);
    sort(nodes.begin(), nodes.end(), [&](int a, int b) {
        if (depth[a] != depth[b]) return depth[a] < depth[b];
        return a < b;
    });

    for (int v : nodes) printf("%d %d\n", parent[v], v);

    ll used = N - 1;
    for (ll t = used; t < T; t++) {
        // leftover budget: harmless repeat of the last connecting edge
        int lastV = nodes.back();
        printf("%d %d\n", parent[lastV], lastV);
    }
    return 0;
}
