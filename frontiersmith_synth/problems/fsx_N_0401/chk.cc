#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Quantum Coupler Interdiction".
//
// Input:  n m s t k P ; then m edges  u v c  (directed u<v, DAG).
// Output: r  then r distinct 1-based edge indices to REMOVE (0<=r<=k).
//
// Objective (minimize):
//   F(R) = sum of removal costs of R
//        + P * ( number of DISTINCT prime hop-lengths L such that a surviving
//                s->t path of exactly L edges still exists ).
// Baseline B = F(empty set) = P * (#prime lengths present originally).  B>0.
// Score (min): sc = min(1000, 100 * B / max(1,F)); ratio = sc/1000.
//   do-nothing -> F=B -> ratio 0.1 ; better interdiction -> higher ratio.
// -----------------------------------------------------------------------------

static const int MAXL = 2200;   // > max vertices (max path length < n <= 2100)

int n, m, s, t, k;
ll P;
vector<int> eu, ev, ec;                 // 1-based edges
vector<vector<int>> inEdges;            // incoming edge ids per vertex
vector<char> isprime;

// compute F given a removed-flag vector over edge ids [1..m]
ll computeF(const vector<char>& removed){
    static vector<bitset<MAXL>> reach;
    reach.assign(n, bitset<MAXL>());
    reach[s].set(0);
    for (int v = 0; v < n; v++){
        if (v == s) continue;
        for (int id : inEdges[v]){
            if (removed[id]) continue;
            reach[v] |= (reach[eu[id]] << 1);
        }
    }
    int primeLengths = 0;
    for (int L = 2; L < n && L < MAXL; L++)
        if (isprime[L] && reach[t].test(L)) primeLengths++;
    ll cost = 0;
    for (int id = 1; id <= m; id++) if (removed[id]) cost += ec[id];
    return cost + P * (ll)primeLengths;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    s = inf.readInt();
    t = inf.readInt();
    k = inf.readInt();
    P = inf.readLong();

    eu.assign(m + 1, 0); ev.assign(m + 1, 0); ec.assign(m + 1, 0);
    inEdges.assign(n, {});
    for (int i = 1; i <= m; i++){
        int u = inf.readInt();
        int v = inf.readInt();
        int c = inf.readInt();
        eu[i] = u; ev[i] = v; ec[i] = c;
        inEdges[v].push_back(i);
    }
    isprime.assign(max(n, MAXL) + 1, 1);
    isprime[0] = isprime[1] = 0;
    for (int i = 2; (ll)i * i <= max(n, MAXL); i++)
        if (isprime[i]) for (int j = i * i; j <= max(n, MAXL); j += i) isprime[j] = 0;

    // ---- read participant output ----
    vector<char> removed(m + 1, 0);
    int r = ouf.readInt(0, k, "r");
    for (int i = 0; i < r; i++){
        int id = ouf.readInt(1, m, "edge_index");
        if (removed[id]) quitf(_wa, "edge index %d listed more than once", id);
        removed[id] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective and baseline ----
    ll F = computeF(removed);
    vector<char> none(m + 1, 0);
    ll B = computeF(none);
    if (B <= 0) B = 1;  // safety; generator guarantees B>0

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld removed=%d Ratio: %.6f", F, B, r, sc / 1000.0);
    return 0;
}
