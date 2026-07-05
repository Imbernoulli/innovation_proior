#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Warehouse Gantry Restacking".
//
// Input:  n p H ; then p start stacks ; then p goal stacks (each: k c_1..c_k,
//         bottom to top). Crates 1..n; positions n..p-1 empty in both.
// Output: M ; then M moves "c q": drop clear crate c onto position q.
//
// Objective (minimize): F = sum over moves of ( |R-a| + |a-q| + H ),
//   where a = crate's current position and R = gantry position (starts at 0).
// Feasibility: each move legal (c clear, q != a), M in [0,30n], final == goal.
// Baseline B = effort of the naive teardown (park every crate to its own staging
//   slot, then rebuild each goal stack bottom-to-top).  B > 0.
// Score (min): sc = min(1000, 100 * B / max(1,F)); ratio = sc/1000.
//   naive teardown -> F = B -> ratio 0.1 ; smarter plans -> higher, capped at 1.0.
// -----------------------------------------------------------------------------

int n, p, H;
vector<vector<int>> startCols, goalCols;

// Effort of the grader's naive teardown, simulated on a fresh copy of the start.
ll baselineEffort(){
    vector<vector<int>> cur = startCols;
    vector<int> colOf(n + 1, -1);
    for (int col = 0; col < p; col++) for (int c : cur[col]) colOf[c] = col;
    ll R = 0, cost = 0;
    auto park = [&](int c){ return p - n + (c - 1); };   // unique empty staging slot
    // disassemble all shelf columns [0..n-1], top to bottom
    for (int pos = 0; pos < n; pos++){
        while (!cur[pos].empty()){
            int c = cur[pos].back(), a = pos, q = park(c);
            cost += llabs(R - a) + llabs((ll)a - q) + H;
            cur[pos].pop_back(); cur[q].push_back(c); colOf[c] = q; R = q;
        }
    }
    // rebuild each goal column bottom to top
    for (int gpos = 0; gpos < n; gpos++){
        for (int c : goalCols[gpos]){
            int a = colOf[c], q = gpos;
            cost += llabs(R - a) + llabs((ll)a - q) + H;
            cur[a].pop_back(); cur[q].push_back(c); colOf[c] = q; R = q;
        }
    }
    return cost;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    n = inf.readInt(); p = inf.readInt(); H = inf.readInt();
    startCols.assign(p, {});
    goalCols.assign(p, {});
    for (int col = 0; col < p; col++){
        int k = inf.readInt(0, n);
        for (int j = 0; j < k; j++) startCols[col].push_back(inf.readInt(1, n));
    }
    for (int col = 0; col < p; col++){
        int k = inf.readInt(0, n);
        for (int j = 0; j < k; j++) goalCols[col].push_back(inf.readInt(1, n));
    }

    // ---- simulate participant plan with strict feasibility validation ----
    vector<vector<int>> cur = startCols;
    vector<int> colOf(n + 1, -1);
    for (int col = 0; col < p; col++) for (int c : cur[col]) colOf[c] = col;
    ll R = 0, F = 0;
    ll MAXM = 30LL * n;

    int M = ouf.readInt(0, (int)MAXM, "M");
    for (int i = 0; i < M; i++){
        int c = ouf.readInt(1, n, "crate");
        int q = ouf.readInt(0, p - 1, "dest");
        int a = colOf[c];
        if (a < 0) quitf(_wa, "move %d: crate %d not present", i + 1, c);
        if (cur[a].empty() || cur[a].back() != c)
            quitf(_wa, "move %d: crate %d is not clear (top of position %d)", i + 1, c, a);
        if (q == a) quitf(_wa, "move %d: destination equals source position %d", i + 1, a);
        F += llabs(R - a) + llabs((ll)a - q) + H;
        cur[a].pop_back(); cur[q].push_back(c); colOf[c] = q; R = q;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    for (int col = 0; col < p; col++)
        if (cur[col] != goalCols[col])
            quitf(_wa, "final configuration differs from goal at position %d", col);

    ll B = baselineEffort();
    if (B <= 0) B = 1;   // safety; generator guarantees B > 0

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld moves=%d Ratio: %.6f", F, B, M, sc / 1000.0);
    return 0;
}
