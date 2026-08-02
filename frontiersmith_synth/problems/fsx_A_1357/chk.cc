#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Tunnel Sweep: Clearing the Underlevel Network".
//
// Input:
//   N
//   (N-1) lines: p_i c_i   for i = 2..N   (p_i < i, so processing node ids in
//   DECREASING order is a valid bottom-up topological order)
//
// Output: for every junction v with >=1 child, one line "v k u_1 ... u_k",
//   a permutation of v's children giving the chosen clearing order.
//
// x_v = 1                                                     if v is a leaf
// x_v = max_i ( x_{u_i} + c_{u_1} + ... + c_{u_{i-1}} )        otherwise,
//   for the participant-chosen order u_1..u_k of v's children.
//
// Objective (MIN): F = x_1.
// Baseline B (checker-computed): x_1 evaluated with the ASCENDING-id order at
//   every junction (exactly what solutions/trivial.cpp prints, so F=B there).
// Score: sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static int N;
static vector<int> par, cst;
static vector<vector<int>> children;   // true children list per node, ascending id

// Evaluate x_1 given, for every internal node v, an explicit clearing order
// order[v] (a permutation of children[v]). Processes ids N..1 (valid bottom-up
// order since par[i] < i for all i).
static ll evalWithOrders(const vector<vector<int>>& order){
    vector<ll> x(N + 1, 0);
    for (int v = N; v >= 1; v--){
        if (children[v].empty()){
            x[v] = 1;
            continue;
        }
        ll run = 0, best = 0;
        for (int u : order[v]){
            best = max(best, x[u] + run);
            run += cst[u];
        }
        x[v] = best;
    }
    return x[1];
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt(2, 3000, "N");
    par.assign(N + 1, 0);
    cst.assign(N + 1, 0);
    children.assign(N + 1, {});
    for (int i = 2; i <= N; i++){
        int p = inf.readInt(1, i - 1, "p_i");
        int c = inf.readInt(1, 1000, "c_i");
        par[i] = p;
        cst[i] = c;
        children[p].push_back(i);
    }
    for (int v = 1; v <= N; v++) sort(children[v].begin(), children[v].end());

    vector<char> isInternal(N + 1, 0);
    int internalCount = 0;
    for (int v = 1; v <= N; v++)
        if (!children[v].empty()){ isInternal[v] = 1; internalCount++; }

    // ---- read participant output: one line per internal junction ----
    vector<char> given(N + 1, 0);
    vector<vector<int>> order(N + 1);
    int linesSeen = 0;

    while (!ouf.seekEof()){
        int v = ouf.readInt(1, N, "v");
        if (!isInternal[v]) quitf(_wa, "junction %d is a leaf (or out of range) -- must not appear in the output", v);
        if (given[v]) quitf(_wa, "junction %d printed more than once", v);
        given[v] = 1;
        linesSeen++;

        int k = ouf.readInt(0, N - 1, "k");
        if ((size_t)k != children[v].size())
            quitf(_wa, "junction %d: declared k=%d but true child count is %d", v, k, (int)children[v].size());

        vector<int> declared(k);
        vector<char> seenLocal(N + 1, 0);
        for (int j = 0; j < k; j++){
            int u = ouf.readInt(2, N, "child id");
            if (par[u] != v) quitf(_wa, "junction %d: listed id %d is not one of its children", v, u);
            if (seenLocal[u]) quitf(_wa, "junction %d: child %d repeated in its clearing order", v, u);
            seenLocal[u] = 1;
            declared[j] = u;
        }
        order[v] = declared;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (linesSeen != internalCount) quitf(_wa, "expected exactly one line per internal junction (%d), got %d", internalCount, linesSeen);
    for (int v = 1; v <= N; v++)
        if (isInternal[v] && !given[v]) quitf(_wa, "missing output line for internal junction %d", v);

    ll F = evalWithOrders(order);

    // ---- internal baseline: ascending-id order at every junction ----
    vector<vector<int>> ascOrder(N + 1);
    for (int v = 1; v <= N; v++) ascOrder[v] = children[v]; // already sorted ascending
    ll B = evalWithOrders(ascOrder);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
