#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for "The Tangled Arcade" (MAXIMIZE).
//
// Each test bundles M independent position instances. Each instance is a directed graph on
// n nodes (0..n-1) with e directed edges and c "active tokens" (current positions of c
// independent components of a disjunctive sum -- the components themselves are never
// labeled; they are provable from the graph structure alone since no edge ever crosses
// between two components by construction).
//
// Ground truth per node (Sprague-Grundy with a CHAOS rule for cycles):
//   - Tarjan-SCC the instance graph, condense to a DAG.
//   - A node whose SCC has size >= 2, or that has a self-loop, is CHAOTIC: by this puzzle's
//     rule a jammed/looping component always contributes grundy 0.
//   - A non-chaotic (singleton, loop-free) node's grundy = mex{ grundy(v) : edge (node,v) }
//     (well-defined: SCC ids are assigned in Tarjan's completion order, which is exactly
//     reverse-topological, so every successor SCC is already resolved).
// Instance value X0 = XOR of grundy(token) over the c active tokens. A move relocates ONE
// token u along a real edge u->v; it is a WINNING move iff X0 ^ grundy(u) ^ grundy(v) == 0.
//
// Participant output: for each instance, one line "PASS" (claiming no winning move exists)
// or "MOVE u v" (a claimed winning move). Score:
//   F = # instances where the claim is correct.
//   B = # instances where "always PASS" (the grader's own trivial construction) is correct,
//       i.e. where no winning move actually exists. B is guaranteed > 0 by planted
//       zero-out-degree instances in every test.
//   ratio = min(1000, 100*F/B) / 1000   (matches trivial exactly when it always outputs PASS)

static const int MAXN_HARD = 4000;

int n_g;
vector<vector<int>> adj_g;
vector<int> comp_id, low_id, disc_id, scc_id;
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

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    long long M = inf.readLong();
    if (M < 1 || M > 100000) quitf(_fail, "generator/input malformed: M=%lld", M);

    long long F = 0, B = 0;

    for (long long m = 0; m < M; m++) {
        int n = inf.readInt();
        int c = inf.readInt();
        int e = inf.readInt();
        if (n < 1 || n > MAXN_HARD || c < 1 || c > n || e < 0 || e > 20000)
            quitf(_fail, "generator/input malformed: instance %lld n=%d c=%d e=%d", m, n, c, e);

        vector<int> tokens(c);
        for (int i = 0; i < c; i++) tokens[i] = inf.readInt(0, n - 1);

        adj_g.assign(n, {});
        selfLoop_g.assign(n, false);
        vector<pair<int,int>> edges(e);
        for (int i = 0; i < e; i++) {
            int u = inf.readInt(0, n - 1), v = inf.readInt(0, n - 1);
            edges[i] = {u, v};
            adj_g[u].push_back(v);
            if (u == v) selfLoop_g[u] = true;
        }

        // ---- Tarjan SCC ----
        disc_id.assign(n, -1); low_id.assign(n, -1); scc_id.assign(n, -1);
        onStk.assign(n, false); stk.clear(); timer_g = 0; sccCnt_g = 0;
        n_g = n;
        for (int i = 0; i < n; i++) if (disc_id[i] == -1) tarjan(i);

        vector<int> sccSize(sccCnt_g, 0);
        for (int i = 0; i < n; i++) sccSize[scc_id[i]]++;
        vector<char> chaoticScc(sccCnt_g, 0);
        for (int s = 0; s < sccCnt_g; s++) if (sccSize[s] >= 2) chaoticScc[s] = 1;
        for (int i = 0; i < n; i++) if (selfLoop_g[i]) chaoticScc[scc_id[i]] = 1;

        // grundy per SCC, processed in increasing id order == Tarjan completion order ==
        // reverse-topological (every out-edge of scc s lands in an scc with a SMALLER id,
        // already resolved).
        vector<int> gsccVal(sccCnt_g, 0);
        for (int s = 0; s < sccCnt_g; s++) {
            if (chaoticScc[s]) { gsccVal[s] = 0; continue; }
            // singleton loop-free scc: find its one member, mex over successor grundies
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

        set<int> tokenSet(tokens.begin(), tokens.end());
        bool W = false;
        for (int tk : tokens) {
            int target = X0 ^ grundy[tk];
            for (int v : adj_g[tk]) {
                if (grundy[v] == target) { W = true; break; }
            }
            if (W) break;
        }
        if (!W) B++;

        // ---- read participant's claim for this instance ----
        string word = ouf.readWord();
        bool instanceOK;
        if (word == "PASS") {
            instanceOK = !W;
        } else if (word == "MOVE") {
            int u = ouf.readInt(0, n - 1, "move-u");
            int v = ouf.readInt(0, n - 1, "move-v");
            if (!tokenSet.count(u))
                quitf(_wa, "instance %lld: MOVE from node %d which is not an active token", m, u);
            bool edgeOk = false;
            for (int w : adj_g[u]) if (w == v) { edgeOk = true; break; }
            if (!edgeOk)
                quitf(_wa, "instance %lld: no edge %d->%d exists", m, u, v);
            int result = X0 ^ grundy[u] ^ grundy[v];
            instanceOK = (result == 0);
        } else {
            quitf(_wa, "instance %lld: expected PASS or MOVE, got '%s'", m, word.c_str());
            return 0; // unreachable
        }
        if (instanceOK) F++;
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing data after %lld instances", M);

    if (B <= 0) quitf(_fail, "generator/input malformed: baseline B=0 (no instance has W=false)");

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
