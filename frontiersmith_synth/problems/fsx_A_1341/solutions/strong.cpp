// TIER: strong
// The insight: this is a disjunctive SUM of independent components hidden by interleaved
// node ids. Rather than search the joint (exponential-in-C) state space, compute each
// node's Grundy value LOCALLY via Tarjan-SCC condensation + mex DP (a node in a nontrivial
// SCC or with a self-loop is CHAOTIC -> forced grundy 0, the rule for a jammed/looping
// component), then combine the c active tokens' grundy values by XOR. A winning move exists
// iff some token u has an edge to a node v with grundy(v) == X0 ^ grundy(u). This is
// O(V+E) per instance, totally insensitive to the number of components C -- turning what
// looks like an exponential joint search into a linear scan.
#include <bits/stdc++.h>
using namespace std;

int n_g;
vector<vector<int>> adj_g;
vector<int> disc_id, low_id, scc_id;
vector<bool> onStk;
vector<int> stk;
int timer_g, sccCnt_g;
vector<bool> selfLoop_g;

void tarjan(int u) {
    disc_id[u] = low_id[u] = timer_g++;
    stk.push_back(u); onStk[u] = true;
    for (int v : adj_g[u]) {
        if (disc_id[v] == -1) {
            tarjan(v);
            low_id[u] = min(low_id[u], low_id[v]);
        } else if (onStk[v]) {
            low_id[u] = min(low_id[u], disc_id[v]);
        }
    }
    if (low_id[u] == disc_id[u]) {
        while (true) {
            int w = stk.back(); stk.pop_back(); onStk[w] = false;
            scc_id[w] = sccCnt_g;
            if (w == u) break;
        }
        sccCnt_g++;
    }
}

int main() {
    long long M;
    if (!(cin >> M)) return 0;
    for (long long m = 0; m < M; m++) {
        int n, c, e;
        cin >> n >> c >> e;
        vector<int> tokens(c);
        for (int i = 0; i < c; i++) cin >> tokens[i];
        adj_g.assign(n, {});
        selfLoop_g.assign(n, false);
        for (int i = 0; i < e; i++) {
            int u, v; cin >> u >> v;
            adj_g[u].push_back(v);
            if (u == v) selfLoop_g[u] = true;
        }

        disc_id.assign(n, -1); low_id.assign(n, -1); scc_id.assign(n, -1);
        onStk.assign(n, false); stk.clear(); timer_g = 0; sccCnt_g = 0;
        n_g = n;
        for (int i = 0; i < n; i++) if (disc_id[i] == -1) tarjan(i);

        vector<int> sccSize(sccCnt_g, 0);
        for (int i = 0; i < n; i++) sccSize[scc_id[i]]++;
        vector<char> chaoticScc(sccCnt_g, 0);
        for (int s = 0; s < sccCnt_g; s++) if (sccSize[s] >= 2) chaoticScc[s] = 1;
        for (int i = 0; i < n; i++) if (selfLoop_g[i]) chaoticScc[scc_id[i]] = 1;

        vector<int> gsccVal(sccCnt_g, 0);
        for (int s = 0; s < sccCnt_g; s++) {
            if (chaoticScc[s]) { gsccVal[s] = 0; continue; }
            int node = -1;
            for (int i = 0; i < n; i++) if (scc_id[i] == s) { node = i; break; }
            vector<int> succ;
            for (int v : adj_g[node]) succ.push_back(gsccVal[scc_id[v]]);
            sort(succ.begin(), succ.end());
            succ.erase(unique(succ.begin(), succ.end()), succ.end());
            int mex = 0;
            for (int x : succ) { if (x == mex) mex++; else if (x > mex) break; }
            gsccVal[s] = mex;
        }
        vector<int> grundy(n);
        for (int i = 0; i < n; i++) grundy[i] = gsccVal[scc_id[i]];

        int X0 = 0;
        for (int tk : tokens) X0 ^= grundy[tk];

        bool found = false;
        int bu = -1, bv = -1;
        for (int tk : tokens) {
            int target = X0 ^ grundy[tk];
            for (int v : adj_g[tk]) {
                if (grundy[v] == target) { found = true; bu = tk; bv = v; break; }
            }
            if (found) break;
        }
        if (found) cout << "MOVE " << bu << " " << bv << "\n";
        else cout << "PASS\n";
    }
    return 0;
}
